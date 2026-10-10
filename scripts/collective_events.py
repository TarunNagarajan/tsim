"""Validate actual collective dispatch groups and virtual source-lane maps."""
from collections import Counter
import re

def collective_kind(instruction):
    plain = re.sub(r'^\s*@\s*!?\s*\S+\s+', '', instruction).strip()
    name = plain.split()[0].split('.')[0]
    return name if name in ['shfl', 'vote', 'redux', 'match'] else None

class CollectiveEvents:
    def __init__(self, issues):
        self.issues = issues
        self.groups = {}
        self.group_ids = {}
        self.masks = {}
        self.waits = set()
        self.counts = Counter()

    def record(self, event):
        try: self._record(event)
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError('Malformed collective event') from error

    def _record(self, event):
        parent = self.issues[event['issue']]
        kind = collective_kind(parent['instruction'])
        if kind is None: raise ValueError('Collective record belongs to an ordinary instruction')
        def integer(value, minimum=0):
            if type(value) is not int or value < minimum: raise ValueError('Invalid collective integer identity')
            return value
        def mask(value):
            if integer(value) >= 1 << 32: raise ValueError('Invalid collective lane mask')
            return value
        record = event['type']; issue = integer(event['issue'], 1)
        if record == 'collective_wait':
            if issue in self.waits or self.masks.get(issue) or not parent['execution_mask']:
                raise ValueError('Invalid or duplicate collective wait')
            self.waits.add(issue); self.counts['wait_attempts'] += 1
        elif record == 'collective_group':
            lanes = mask(event['execution_mask'])
            if not lanes or lanes & ~parent['execution_mask'] or lanes & self.masks.get(issue, 0) or issue in self.waits:
                raise ValueError('Collective groups must partition executing lanes')
            if event['kind'] != kind: raise ValueError('Collective kind differs from parent instruction')
            source = integer(event['data_use'], -1)
            if source != -1 and (not parent['uses'] or source != parent['uses'][0]):
                raise ValueError('Collective data source differs from first explicit operand')
            defs = event['data_defs']
            if not isinstance(defs, list) or any(type(key) is not int or key not in parent['defs'] for key in defs) or len(set(defs)) != len(defs):
                raise ValueError('Invalid collective data destinations')
            if kind != 'shfl' and defs != parent['defs']:
                raise ValueError('Collective data destinations differ from parent')
            if kind == 'shfl':
                plain = re.sub(r'^\s*@\s*!?\s*\S+\s+', '', parent['instruction']).strip()
                destination = plain.split(None, 1)[1].split(',')[0].split('|')[0].strip()
                expected = [] if destination == '_' else parent['defs'][:1]
                if defs != expected: raise ValueError('Shuffle predicate does not depend on transferred data')
            sequence = integer(event['sequence'], 1)
            if sequence in self.groups: raise ValueError('Duplicate collective group identity')
            self.groups[sequence] = {'event': event, 'lanes': set()}
            self.group_ids.setdefault(issue, []).append(sequence)
            self.masks[issue] = self.masks.get(issue, 0) | lanes
            self.counts[kind + '_groups'] += 1
        elif record == 'collective_transfer':
            group = self.groups[integer(event['group'], 1)]
            dispatch = group['event']; destination = integer(event['destination_lane'])
            sources = mask(event['source_mask'])
            if dispatch['issue'] != issue or dispatch['data_use'] < 0 or not dispatch['data_defs']:
                raise ValueError('Collective transfer lacks virtual source/destination ownership')
            if destination >= 32 or not dispatch['execution_mask'] & (1 << destination) or destination in group['lanes']:
                raise ValueError('Invalid or duplicate collective destination lane')
            if sources & ~dispatch['execution_mask']: raise ValueError('Collective source lies outside executed group')
            if kind == 'shfl':
                if sources.bit_count() > 1: raise ValueError('Shuffle must read at most one source lane')
                if not sources: self.counts['undefined_shuffle_sources'] += 1
            elif sources != dispatch['execution_mask']:
                raise ValueError('Reduction/vote/match requires all contributing source lanes')
            group['lanes'].add(destination)
            self.counts['destination_maps'] += 1
            self.counts['source_lane_reads'] += sources.bit_count()
        else: raise ValueError('Unknown collective record')

    def finish(self, issue, status):
        parent = self.issues[issue]
        if collective_kind(parent['instruction']) is None or status != 'completed': return
        if issue not in self.waits and self.masks.get(issue, 0) != parent['execution_mask']:
            raise ValueError('Collective dispatch participation conservation failed')
        for identity in self.group_ids.get(issue, []):
            group = self.groups[identity]
            event = group['event']
            expected = event['execution_mask'].bit_count() if event['data_use'] >= 0 and event['data_defs'] else 0
            if len(group['lanes']) != expected:
                raise ValueError('Collective source-map conservation failed')
        for identity in self.group_ids.pop(issue, []): self.groups.pop(identity)
        self.masks.pop(issue, None); self.waits.discard(issue)
