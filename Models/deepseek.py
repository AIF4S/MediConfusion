import torch
from transformers import AutoModelForCausalLM

def split_model(model_name):
    device_map = {}
    model_splits = {
        'deepseek-ai/deepseek-vl2-small': [13, 14], # 2 GPU for 16b
        'deepseek-ai/deepseek-vl2': [10, 10, 10], # 3 GPU for 27b
    }
    num_layers_per_gpu = model_splits[model_name]
    num_layers =  sum(num_layers_per_gpu)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision'] = 0
    device_map['projector'] = 0
    device_map['image_newline'] = 0
    device_map['view_seperator'] = 0
    device_map['language.model.embed_tokens'] = 0
    device_map['language.model.norm'] = 0
    device_map['language.lm_head'] = 0
    device_map[f'language.model.layers.{num_layers - 1}'] = 0
    return device_map

def load_model(model_path):
    if 'janus' in model_path:
        from janus.models import MultiModalityCausalLM, VLChatProcessor
        global load_pil_images
        from janus.utils.io import load_pil_images
        vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        janus = True

    elif 'deepseek-vl2' in model_path:
        from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
        global load_pil_images
        from deepseek_vl2.utils.io import load_pil_images
        device_map = split_model(model_path)
        vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
        vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map=device_map)
        janus = False

    tokenizer = vl_chat_processor.tokenizer
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    # vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    return vl_gpt, vl_chat_processor, tokenizer, janus

def ask_question(model, processor, tokenizer, image_path, question, temperature, max_new_tokens=512, do_sample=False, janus=False):
    if janus:
        conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [image_path],
        },
        {"role": "<|Assistant|>", "content": ""},
        ]
    else:
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image>\n{question}",
                "images": [image_path],
            },
            {"role": "<|Assistant|>", "content": ""},
            ]

    pil_images = load_pil_images(conversation)
    prepare_inputs = processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(model.device)

    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
    
    if janus:
        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            use_cache=True,
        )
    else:
        outputs = model.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            use_cache=True,
        )

    response = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return response