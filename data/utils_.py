
import requests
from PIL import Image
from io import BytesIO

def load_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Check if the request was successful
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"Error loading image from {url}: {e}")
        return None


def find_longest_common_prefix(strings):
    if not strings:
        return ""

    # Find the minimum length among all strings
    min_length = min(len(s) for s in strings)

    # Find the common prefix
    common_prefix = ""
    for i in range(min_length):
        # Check if all strings have the same character at position i
        char = strings[0][i]
        if all(s[i] == char for s in strings):
            common_prefix += char
        else:
            break
    
    return common_prefix


def cleanup_description(category, description, broad_category_name):
    # remove last . from category
    category = category.rstrip('.')
    if broad_category_name is None or broad_category_name == "" or broad_category_name == "None":
        broad_category_name = category

    if category in description:
        if f'the {category} ' in description:
            self_ref_description = description.replace(category, 'item')
        elif f'an {category} ' in description:
            self_ref_description = description.replace(f'an {category}', 'the item')
        else:
            self_ref_description = description.replace(category, 'the item')
    elif broad_category_name in description:
        if f'the {broad_category_name} ' in description:
            self_ref_description = description.replace(broad_category_name, 'item')
        elif f'an {broad_category_name} ' in description:
            self_ref_description = description.replace(f'an {broad_category_name}', 'the item')
        else:
            self_ref_description = description.replace(broad_category_name, 'the item')
    else:
        self_ref_description = description

    if category.startswith('an '):
        category = category[3:]
    elif category.startswith('a '):
        category = category[2:]
    
    if self_ref_description.startswith('an '):
        self_ref_description = self_ref_description[3:]
    elif self_ref_description.startswith('a '):
        self_ref_description = self_ref_description[2:]
    # broad_category_name=category #item["description"]["category"] 
    # # if broad_category_name is None or broad_category_name == "" or broad_category_name == "None":
    #     # broad_category_name = category
    full_description = description.replace(' it ', f' {category} ')
    full_description = full_description.replace(' this item ', f' {category} ')
    full_description = full_description.replace(' this object ', f' {category} ')
    return category, full_description, self_ref_description


CANONICAL_EDIT_TYPES = [
    "local color change", "texture", "shape change", "insertion", "remove",
    "replace", "background change", "style", "action", "geometric", "text",
]

def match_edit_type(edit_type):
    """Map raw edit_type strings to canonical types.

    Returns None for samples that should be skipped (e.g. multi-type with remove).
    """
    edit_type = edit_type.lower().strip()

    # --- Single-type exact mapping ---
    single_mapping = {
        'adjust (shape change)': 'shape change',
        'shape change': 'shape change',
        'shape': 'shape change',
        'removal': 'remove',
        'remove': 'remove',
        'bg': 'background change',
        'background change': 'background change',
        'background_swap': 'background change',
        'add': 'insertion',
        'addition': 'insertion',
        'insertion': 'insertion',
        'local texture': 'texture',
        'texture': 'texture',
        'local color change': 'local color change',
        'change_color': 'local color change',
        'change_local': 'local color change',
        'object_swap': 'replace',
        'swap': 'replace',
        'style': 'style',
        'replace': 'replace',
        'replacement': 'replace',
        'action': 'action',
        'geometric': 'geometric',
        'text': 'text',
        # GPT metadata fallback task types
        'attribute_modification': 'replace',
        'env': 'background change',
        'edit': 'replace',
        'complex-edit': 'complex-edit',
        'change_global': 'background change',
        'transform_global': 'replace',
        'transform_local': 'texture',
        'turn': 'replace',
        'others': 'other',
        # Rare PicoBanana types
        'outpainting': 'geometric',
        'relocation': 'geometric',
        'local text change': 'text',
        'local text edit': 'text',
        'local text addition': 'text',
        'local text replacement': 'text',
        'local text translation': 'text',
        'text replacement': 'text',
        'text translation': 'text',
        'local lighting change': 'local color change',
        'local lighting': 'local color change',
        'local weather effect': 'style',
        'local weather and atmosphere change': 'style',
        'clothing edit': 'replace',
        'change age / gender': 'replace',
        'pose change': 'action',
        'local expression change': 'action',
        'skip': None,
    }

    if edit_type in single_mapping:
        return single_mapping[edit_type]

    # --- Multi-type labels (comma or " and " separated) ---
    if ',' in edit_type or ' and ' in edit_type:
        parts = [p.strip() for p in edit_type.replace(' and ', ', ').split(',') if p.strip()]
        # Skip if removal is one of the sub-types
        canonical_parts = set()
        for p in parts:
            c = single_mapping.get(p, p)
            if c is None:
                return None
            canonical_parts.add(c)
        if 'removal' in canonical_parts:
            return None
        return 'complex-edit'

    # Fallback: return as-is (will be dropped if not in edit_type_probs)
    return edit_type
import importlib

def get_style_subtype(edit_instruction):
    if 'watercolor' in edit_instruction:
        return 'watercolor'
    elif 'vintage' in edit_instruction:
        return 'vintage'
    elif 'gradient' in edit_instruction:
        return 'gradient'
    elif 'retro' in edit_instruction:
        return 'retro'
    else:
        return 'other'

def get_local_texture_subtype(edit_instruction):
    if 'glossy' in edit_instruction:
        return 'gloss'
    elif 'shiny' in edit_instruction:
        return 'shiny'
    elif 'shimmer' in edit_instruction:
        return 'shimmer'
    elif 'marble' in edit_instruction:
        return 'marble'
    elif 'metallic' in edit_instruction:
        return 'metallic'
    elif 'smooth' in edit_instruction:
        return 'smooth'
    elif 'dew' in edit_instruction:
        return 'dew'
    elif 'wet' in edit_instruction:
        return 'wet'
    elif 'rough' in edit_instruction:
        return 'rough'
    else:
        return 'other'
    
from collections import defaultdict
def get_data(keys, data, num_samples_ratio=1.0):
    new_data = defaultdict(list)
    counter = defaultdict(int)
    counter_sub_type = defaultdict(lambda: defaultdict(int))
    data_sub_type = defaultdict(lambda: defaultdict(list))
    
    for i, key in enumerate(keys):
        value = data[key]
        for item in value:
            edit_type = match_edit_type(item['edit_type'])
            if edit_type not in ['shape change', 'style', 'bg', 'remove', 'add', 'replace', 'action', 'local color change', 'local texture', 'text']:
                continue
            edit_instruction = item['edit_instruction']
            edited_caption = item['edited_caption']
            if edit_type == 'style':
                subtype = get_style_subtype(edit_instruction)
                counter_sub_type[edit_type][subtype] += 1
                data_sub_type[edit_type][subtype].append(len(new_data[edit_type])-1)
                
            if edit_type == 'local texture':
                subtype = get_local_texture_subtype(edit_instruction)
                counter_sub_type[edit_type][subtype] += 1
                data_sub_type[edit_type][subtype].append(len(new_data[edit_type])-1)
            counter[edit_type] += 1
            new_data[edit_type].append({
                'edit_instruction': edit_instruction,
                'edited_caption': edited_caption,
                'strImagehash': key,
                'questions': item['questions'] if 'questions' in item else None,
            })
            
    return new_data, counter, counter_sub_type, data_sub_type


def check_painting_art(merged_tags, caption):
    if 'painting' in merged_tags or 'art' in merged_tags or 'render' in merged_tags or 'illustration' in merged_tags or 'drawing' in merged_tags or 'sketch' in merged_tags:
        return True
    if 'painting' in caption or 'art' in caption or 'render' in caption or 'illustration' in caption or 'drawing' in caption or 'sketch' in caption:
        return True
    return False