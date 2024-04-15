import sys

import json
from docx import Document
import xml.etree.ElementTree as ET
from pathlib import Path


def write_docx(data, output_file):
    document = Document()

    location = "Lethbridge"
    interviewer = "善積"
    date = "[date]"
    speakers = [
        "L000/佐藤さん"
    ]

    time = 1092000

    speaker_string = ", ".join(f"[{speaker}]" for speaker in speakers)

    time //= 1000
    minutes, seconds = time // 60, time % 60
    hours, minutes = minutes // 60, minutes % 60

    time_string = ""
    if hours:
        time_string += f"{hours}時間"
    if minutes or hours:
        time_string += f"{minutes}分"
    if seconds or minutes or hours:
        time_string += f"{seconds}秒"

    document.add_paragraph(f"""{location}, {date}, {speaker_string}, [1 {interviewer}]
    録音時間 {time_string}
    """)

    text_paragraph = document.add_paragraph()
    document.add_page_break()
    gloss_paragraph = document.add_paragraph()

    for chunk in data:
        speaker = chunk['speaker']
        text = chunk['text']
        try:
            gloss = chunk['gloss']
        except KeyError:
            gloss = "placeholder"
        
        text_paragraph.add_run(f"[{speaker}] {text} ")
        gloss_paragraph.add_run(f"[{speaker}] {gloss} ")

    document.save(output_file)


def get_time_index(time):
    global times
    if time == -1:
        times = []
        return

    if time in times:
        return times.index(time) + 1

    times.append(time)
    return len(times)


def write_xml(data, output_file, media_path):
    root = ET.Element('ANNOTATION_DOCUMENT', {
        'AUTHOR':                           '',
        'DATE':                             '2023-12-11T07:52:21-05:00',
        'FORMAT':                           '3.0',
        'VERSION':                          '3.0',
        'xmlns:xsi':                        'http://www.w3.org/2001/XMLSchema-instance',
        'xsi:noNamespaceSchemaLocation':    'http://www.mpi.nl/tools/elan/EAFv3.0.xsd'
    })

    header = ET.SubElement(root, 'HEADER', {
        'MEDIA_FILE':   str(media_path),
        'TIME_UNITS':   'milliseconds'
    })

    time_order = ET.SubElement(root, 'TIME_ORDER')

    tiers = dict()

    speakers = [0, 1]

    for speaker in speakers:
        speaker_str = str(speaker)

        tiers[f'tx@{speaker}'] = ET.SubElement(root, 'TIER', {
            'LINGUISTIC_TYPE_REF':  'default-lt',
            'PARTICIPANT':          speaker_str,
            'TIER_ID':              f'tx@{speaker_str}',
        })
        
        tiers[f'gl@{speaker}'] = ET.SubElement(root, 'TIER', {
            'LINGUISTIC_TYPE_REF':  'translation',
            'PARTICIPANT':          speaker_str,
            'PARENT_REF':           f'tx@{speaker_str}',
            'TIER_ID':              f'gl@{speaker_str}',
        })
        
        tiers[f'note@{speaker}'] = ET.SubElement(root, 'TIER', {
            'LINGUISTIC_TYPE_REF':  'translation',
            'PARTICIPANT':          speaker_str,
            'PARENT_REF':           f'tx@{speaker_str}',
            'TIER_ID':              f'note@{speaker_str}',
        })

    tree = ET.ElementTree(root)

    lingustic_type = ET.SubElement(root, 'LINGUISTIC_TYPE', {
        'GRAPHIC_REFERENCES': "false",
        'LINGUISTIC_TYPE_ID': "default-lt",
        'TIME_ALIGNABLE': "true"
    })

    lingustic_type = ET.SubElement(root, 'LINGUISTIC_TYPE', {
        'CONSTRAINTS': "Symbolic_Association",
        'GRAPHIC_REFERENCES': "false",
        'LINGUISTIC_TYPE_ID': "translation",
        'TIME_ALIGNABLE': "false"
    })

    CONSTRAINTS = [
        ("Time_Subdivision", "Time subdivision of parent annotation's time interval, no time gaps allowed within this interval"),
        ("Symbolic_Subdivision", "Symbolic subdivision of a parent annotation. Annotations refering to the same parent are ordered"),
        ("Symbolic_Association", "1-1 association with a parent annotation"),
        ("Included_In", "Time alignable annotations within the parent annotation's time interval, gaps are allowed")
    ]

    for stereotype, description in CONSTRAINTS:
        ET.SubElement(root, 'CONSTRAINT', {
            'DESCRIPTION': description,
            'STEREOTYPE': stereotype
        })

    get_time_index(-1)

    annotation_count = 0
    for chunk in data:
        start_index = get_time_index(chunk['start'])
        end_index = get_time_index(chunk['end'])
        
        speaker = chunk['speaker']
        if speaker not in speakers: # TODO: remove
            speaker = 0
        
        assert 'text' in chunk
        
        if 'text' in chunk:
            annotation_count += 1
            tx_annotation_id = f"a{annotation_count}"
            
            container = ET.SubElement(tiers[f'tx@{speaker}'], 'ANNOTATION')
            annotation = ET.SubElement(container, 'ALIGNABLE_ANNOTATION', {
                'ANNOTATION_ID': tx_annotation_id,
                'TIME_SLOT_REF1': f"ts{start_index}",
                'TIME_SLOT_REF2': f"ts{end_index}",
            })
            value = ET.SubElement(annotation, 'ANNOTATION_VALUE')
            value.text = chunk['text']

        if 'gloss' in chunk:
            annotation_count += 1

            container = ET.SubElement(tiers[f'tx@{speaker}'], 'ANNOTATION')
            annotation = ET.SubElement(container, 'REF_ANNOTATION', {
                'ANNOTATION_ID': f"a{annotation_count}",
                'ANNOTATION_REF': tx_annotation_id,
            })
            value = ET.SubElement(annotation, 'ANNOTATION_VALUE')
            value.text = chunk['gloss']


    for index, time in enumerate(times, start=1):
        ET.SubElement(time_order, 'TIME_SLOT', {
            'TIME_SLOT_ID': f"ts{index}",
            'TIME_VALUE': str(int(time * 1000))
        })

    ET.indent(tree, space='\t', level=0)
    tree.write(output_file, encoding='utf-8')


def data_from_xml(input_file):
    tree = ET.parse(input_file)
    root = tree.getroot()

    time_dict = {}
    times = root.findall('TIME_ORDER')[0]
    for time_slot in times:
        time_dict[time_slot.get('TIME_SLOT_ID')] = int(time_slot.get('TIME_VALUE')) / 1000

    tiers = root.findall('TIER')
    text, gloss, note = tiers[::3], tiers[1::3], tiers[2::3]

    text_refs = []
    text_dict = {}

    for text_tier, gloss_tier in zip(text, gloss):
        speaker_id = text_tier.get('TIER_ID').split('@', maxsplit=1)[1]

        for annotation in text_tier:
            annotation = annotation[0]

            annotation_id = annotation.get('ANNOTATION_ID')
            start, end = (
                time_dict[annotation.get('TIME_SLOT_REF1')],
                time_dict[annotation.get('TIME_SLOT_REF2')]
            )

            text_refs.append(annotation_id)
            text_dict[annotation_id] = {
                'speaker': int(speaker_id),    # TODO: change?
                'text': annotation[0].text,
                'start': start, 'end': end
                }

        for annotation in gloss_tier:
            annotation = annotation[0]
            text_dict[annotation.get('ANNOTATION_REF')]['gloss'] = annotation[0].text
        
    data = list(text_dict.values())
    data = sorted(data, key=lambda annotation: annotation['start'])

    return data


def main(directory, do_xml=False, do_docx=False):
    assert do_xml or do_docx

    for path in Path(directory).glob("*.json"):
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)

            if do_xml:
                write_xml(data, path.with_suffix('.eaf'), path.with_suffix(''))
            if do_docx:
                write_docx(data, path.with_suffix('.docx'))


if __name__ == '__main__':
    option, directory = sys.argv[1], sys.argv[2]
    match option:
        case 'xml':
            main(directory, do_xml=True)
        case 'docx':
            # main(directory, do_docx=True)
            for path in Path(directory).glob("*.eaf"):
                with open(path, 'r', encoding='utf-8') as file:
                    data = data_from_xml(file)
                    write_docx(data, path.with_suffix('.docx'))
                with open(path.with_suffix('.json'), 'w', encoding='utf-8') as file:
                    json.dump(data, file)
