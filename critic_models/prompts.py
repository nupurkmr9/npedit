eval_prompt = {
"default":{
"question": '''You are a professional digital artist and an expert image editor. You will be provided with two images.

The first being the original real image and the second being an edited version of the first.
The objective is to evaluate if the editing instruction has been executed in the second image.

Editing instruction: <instruction>

Answer with a Yes or No.
Note that sometimes the two images might look identical due to the failure of image editing. Answer No in that case.''',
"answer": "Yes",
"single_image": False
},

"remove":{
"question": '''You are a professional digital artist and an expert image captioner. You will be provided with an image.

Answer with a Yes or No if the image has <object_name>.''',
"answer": "No",
"single_image": True
}
}


eval_prompt_identity = {
"default": {"question": '''You are a professional digital artist and an expert image editor. You will be provided with two images.

Answer with a Yes or No if the second image is the same as the first image on all attributes, except those specified by the edit: <instruction>''',
"answer": "Yes",
"single_image": False},

"remove": {"question": '''You are a professional digital artist and an expert in image editing. You will be provided with two images.

Answer with a Yes or No if the second image is exactly the same as the first image, including the spatial layout. IGNORE the presence or absence of <object_name>.''',
"answer": "Yes",
"single_image": False},

}




SELECTION_PROMPT = {
                    "editing": eval_prompt,
                    "editing_identity": eval_prompt_identity,
                    }