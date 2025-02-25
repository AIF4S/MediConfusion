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
        vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
        janus = True

    elif 'deepseek-vl2' in model_path:
        from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
        global load_pil_images
        from deepseek_vl2.utils.io import load_pil_images
        device_map = split_model(model_path)
        vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
        vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map=device_map)
        vl_gpt = vl_gpt.to(torch.bfloat16).eval()
        janus = False

    tokenizer = vl_chat_processor.tokenizer

    # vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    return vl_gpt, vl_chat_processor, tokenizer, janus

def ask_question(model, processor, tokenizer, image_path, question, temperature, mode, max_new_tokens=512, do_sample=False, janus=False):
    if janus:
        gen_model = model.language_model
        content_tmp = "<image_placeholder>\n{}"
    else:
        gen_model = model.language
        content_tmp = "<image>\n{}"
    if mode == 'prefix':
        return do_prefix_forward(model, gen_model, question, image_path, processor, tokenizer, content_tmp)
    conversation = [
    {
        "role": "<|User|>",
        "content": content_tmp.format(question),
        "images": [image_path],
    },
    {"role": "<|Assistant|>", "content": ""},
    ]

    pil_images = load_pil_images(conversation)
    prepare_inputs = processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(model.device)

    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
    
    if mode == 'greedy':
        return do_forward(gen_model, inputs_embeds, prepare_inputs.attention_mask, tokenizer)
    elif mode in ['mc', 'gpt4']:
        return do_generation(gen_model, inputs_embeds, prepare_inputs, tokenizer, max_new_tokens, do_sample, temperature)


def do_generation(model, inputs_embeds, prepare_inputs, tokenizer, max_new_tokens, do_sample, temperature):
    if temperature > 0:
        do_sample = True
    outputs = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        use_cache=True,
        temperature=temperature,
    )
    response = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return response


def do_forward(model, inputs_embeds, attention_mask, tokenizer):
    VALID_ANSWERS = ['A', 'B']
    TOKEN_IDs = [tokenizer.encode(x, return_tensors="pt", add_special_tokens=False) for x in VALID_ANSWERS]

    with torch.inference_mode():
        out = model.forward(inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,)
        
    logits = out.logits[0, -1, :]
    soft_max = torch.nn.Softmax(dim=0)
    probs = soft_max(torch.cat([logits[x] for x in TOKEN_IDs]))
    outputs = VALID_ANSWERS[probs.argmax().item()]
    return outputs


@torch.no_grad()
def do_prefix_forward(model, gen_model, problem, image, processor, tokenizer, content_tmp):
    # PREFIX_PROMPT_TEMPLATE = "Question: {} Answer: {}"
    device = model.device
    PREFIX_PROMPT_TEMPLATE = problem.get('format')
    scores = []

    qs = problem["question"]

    for option in [problem["option_A"], problem["option_B"]]:
        # prompt = PREFIX_PROMPT_TEMPLATE.format(qs, option)

        conversation = [
        {
            "role": "<|User|>",
            "content": content_tmp.format(qs),
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": f"{option}"},
        ]
        pil_images = load_pil_images(conversation)
        inputs = processor(conversations=conversation, images=pil_images, force_batchify=True).to(model.device)
        inputs_embeds = model.prepare_inputs_embeds(**inputs)

        answer_tokens = tokenizer.encode(option, add_special_tokens=False)
        num_answer_tokens = len(answer_tokens)
        input_ids = inputs["input_ids"]
        # try to find the answer tokens in input ids
        start_indices = []
        for i in range(input_ids.size(1) - num_answer_tokens + 1):
            if torch.equal(input_ids[0, i:i+num_answer_tokens], torch.tensor(answer_tokens).to(device=device)):
                start_indices.append(i)
        
        if len(start_indices) == 0:
            raise ValueError("Answer tokens not found in input_ids")
        answer_start = start_indices[-1]
        answer_start_from_back = answer_start - input_ids.size(1)
        with torch.inference_mode():
            # out = model(**inputs)
            out = gen_model.forward(inputs_embeds=inputs_embeds, attention_mask=inputs.attention_mask)
            # shift by 1 compared to input
            logits = out.logits[0, answer_start_from_back-1:answer_start_from_back-1+num_answer_tokens]
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # Pick the probabilities corresponding to each of the answer tokens
            probs = torch.gather(probs, 1, torch.tensor(answer_tokens).to(device=device).unsqueeze(0))
            prefix_score = torch.prod(probs.pow(1/num_answer_tokens))
            scores.append(prefix_score.item())

    outputs = "A" if scores[0] > scores[1] else "B"
    return outputs