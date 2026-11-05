"""Static two-bus Bergen model derived from n45_2025."""

from tops.ps_models import n45_2025



def _filter_rows(table, field, values):
    header = table[0]
    field_idx = header.index(field)
    return [header] + [row for row in table[1:] if row[field_idx] in values]



def load():
    source = n45_2025.load()
    buses = {'5510', '5560'}
    governors = {}
    for governor_type, table in source.get('gov', {}).items():
        filtered = _filter_rows(table, 'gen', {
            'G5510-1', 'G5510-2', 'G5560-1'})
        if len(filtered) > 1:
            governors[governor_type] = filtered

    return {
        'base_mva': source['base_mva'],
        'f': source['f'],
        'slack_bus': '5510',
        'buses': _filter_rows(source['buses'], 'name', buses),
        'transformers': _filter_rows(source['transformers'], 'name', {'T5510-5560'}),
        'loads': _filter_rows(source['loads'], 'bus', buses),
        'generators': {
            'GEN': _filter_rows(source['generators']['GEN'], 'bus', buses),
        },
        'gov': governors,
    }
