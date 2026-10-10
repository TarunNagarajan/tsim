"""Independent boundary cases and resource admission conservation."""
from dataclasses import replace
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from sm_residency import SMResidency, block_footprint, allocation_footprint, profile_for, resource_ceiling

def verify_residency_components():
    checks = []
    def check(name, condition):
        if not condition: raise AssertionError(name)
        checks.append(name)
    def reject(name, action):
        try: action()
        except ValueError: checks.append(name)
        else: raise AssertionError(name)
    a100, h100 = profile_for('nvidia/a100'), profile_for('nvidia/h100')
    def fp(threads=128, regs=16, shared=0, p=a100):
        return block_footprint(p, [threads, 1, 1], regs, dynamic_shared=shared)
    one = fp(1)
    check('partial warp reserves full register row', one.warps == 1 and one.registers == 512)
    check('no user shared still reserves system bytes', one.shared_bytes == 1024)
    check('shared allocation rounds upward', fp(shared=1).shared_bytes == 1152)
    check('register allocation rounds by warp', fp(regs=17).registers_per_warp == 768)
    three = fp(96, 200)
    check('per-block admission rounds warps to partitions', three.registers == 19200 and three.registers_admission_check == 25600)
    check('public partial-warp block ceiling', resource_ceiling(a100, one)['blocks_per_sm'] == 32 and
          resource_ceiling(a100, one)['warp_occupancy_fraction'] == 0.5 and resource_ceiling(a100, one)['active_thread_fraction'] == 1 / 64)
    check('full thread ceiling', resource_ceiling(a100, fp(1024, 16))['blocks_per_sm'] == 2)
    check('register partition ceiling differs from aggregate floor', resource_ceiling(a100, fp(128, 56))['blocks_per_sm'] == 9)
    check('overfull per-block register file cannot admit', resource_ceiling(a100, fp(1024, 72))['blocks_per_sm'] == 0)
    check('A100 large shared capacity', resource_ceiling(a100, fp(shared=80 * 1024))['blocks_per_sm'] == 2)
    check('H100 larger shared capacity', resource_ceiling(h100, fp(shared=80 * 1024, p=h100))['blocks_per_sm'] == 2 and
          resource_ceiling(h100, fp(shared=60 * 1024, p=h100))['blocks_per_sm'] == 3)
    check('maximum A100 user shared plus reservation fills SM', fp(shared=163 * 1024).shared_bytes == a100.shared_bytes)
    for name, action in [
        ('zero block', lambda: fp(0)), ('oversized block', lambda: fp(1025)),
        ('negative registers', lambda: fp(regs=-1)), ('unresolved spill estimate', lambda: fp(regs=264)),
        ('oversized user shared', lambda: fp(shared=163 * 1024 + 1)),
        ('boolean block', lambda: fp(True)), ('float registers', lambda: fp(regs=8.0)),
        ('negative shared', lambda: fp(shared=-1)), ('unknown GPU', lambda: profile_for('nvidia/unknown')),
        ('large Z dimension', lambda: block_footprint(a100, [1, 1, 65], 8)),
        ('high static shared', lambda: block_footprint(a100, [32, 1, 1], 8, static_shared=49 * 1024)),
        ('altered hardware profile', lambda: block_footprint(replace(a100, warps=128), [32, 1, 1], 8)),
        ('forged allocation footprint', lambda: resource_ceiling(a100, replace(one, registers=0)))]:
        reject(name + ' rejected', action)
    sm = SMResidency(a100)
    large = fp(128, 56)
    for i in range(9): check('register-limited block admitted ' + str(i), sm.admit(i, large))
    full = sm.snapshot()
    check('admission failure is atomic', not sm.admit(9, large) and sm.snapshot() == full)
    reject('duplicate identity rejected', lambda: sm.admit(0, large))
    sm.release(4)
    check('released partition resources can be reused', sm.admit(9, large))
    for identity in list(sm.live): sm.release(identity)
    check('complete release conserves every resource', sm.snapshot() == {'blocks': 0, 'warps': 0, 'threads': 0, 'shared_bytes': 0, 'barriers': 0, 'partition_registers': [0] * 4})
    reject('double release rejected', lambda: sm.release(0))
    rng = random.Random(457)
    for step in range(300):
        if sm.live and rng.randrange(3) == 0: sm.release(rng.choice(list(sm.live)))
        else: sm.admit(step, fp(rng.choice([32, 64, 96, 128, 192, 256]), rng.choice([8, 24, 56, 96, 200]), rng.choice([0, 4097, 16384, 40000])))
        snapshot = sm.snapshot()
        if not (snapshot['warps'] <= 64 and snapshot['blocks'] <= 32 and snapshot['shared_bytes'] <= a100.shared_bytes and
                all(0 <= n <= 16384 for n in snapshot['partition_registers']) and
                sum(snapshot['partition_registers']) == sum(f.registers for f, _ in sm.live.values()) and
                snapshot['shared_bytes'] == sum(f.shared_bytes for f, _ in sm.live.values())):
            raise AssertionError('mixed admission conservation step ' + str(step))
    check('300 mixed admission/release steps conserve budgets', True)
    prefix = allocation_footprint(h100, [32, 1, 1], 12, 16, 124, 1)
    check('compiler prefix preserves padding and charges reservation once', prefix.shared_user_bytes == 140 and prefix.shared_bytes == 1280)
    many = block_footprint(h100, [32, 1, 1], 12, barriers=16)
    check('Hopper barrier-limited ceiling', resource_ceiling(h100, many)['blocks_per_sm'] == 4 and
          resource_ceiling(h100, many)['limiting_resources'] == ['barriers'])
    none = block_footprint(h100, [32, 1, 1], 12, barriers=0)
    check('no barrier kernel has no artificial barrier ceiling', resource_ceiling(h100, none)['blocks_per_sm'] == 32)
    reject('negative barriers rejected', lambda: block_footprint(h100, [32, 1, 1], 12, barriers=-1))
    reject('too many block barriers rejected', lambda: block_footprint(h100, [32, 1, 1], 12, barriers=17))
    pool = SMResidency(h100)
    for identity in range(4):
        if not pool.admit(identity, many): raise AssertionError('barrier capacity under-admission')
    before = pool.snapshot()
    check('barrier admission refusal is atomic', not pool.admit(4, many) and pool.snapshot() == before and before['barriers'] == 64)
    pool.release(1)
    check('released barriers can be reused', pool.admit(4, many) and pool.snapshot()['barriers'] == 64)
    for identity in list(pool.live): pool.release(identity)
    check('barrier allocations completely returned', pool.snapshot()['barriers'] == 0)
    return checks

if __name__ == '__main__':
    print('PASS residency components checks=' + str(len(verify_residency_components())))
