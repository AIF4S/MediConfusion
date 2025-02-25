import os
import base64
from openai import AzureOpenAI
from dotenv import load_dotenv

def get_client(max_retries=2, timeout=20, api_version="2025-01-01-preview", override=True):
    load_dotenv(override=override)
    client = AzureOpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                         api_version=api_version,
                         azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
                         max_retries=max_retries,
                         timeout=timeout,
                         )

    return client

def get_response(client, deployment_name, init_prompt, prompt, temperature, max_retry=3, print_error=True):
    counter = max_retry
    response = None
    while counter > 0:
        try:
            if deployment_name == 'o1':
                response = client.chat.completions.create(model=deployment_name,
                                                        messages=[
                                                            {"role": "user", "content": prompt},
                                                            ],
                                                        )
            else:
                response = client.chat.completions.create(model=deployment_name,
                                                        messages=[
                                                            {"role": "system", "content": init_prompt},
                                                            {"role": "user", "content": prompt},
                                                            ],
                                                        temperature=temperature,
                                                        )
            response = response.choices[0].message.content
            break
        except Exception as e:
            if print_error:
                print(e)
            counter -= 1
    return response

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
 

def ask_question(client, image_path, question, init_prompt, deployment_name, temperature):
    base64_image = encode_image(image_path)
    content = [
        {"type": "text",
            "text": question
        },
        {"type": "image_url",
            "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
    ]
    response = get_response(client=client,
                                deployment_name=deployment_name, 
                                init_prompt=init_prompt, 
                                prompt=content, 
                                temperature=temperature)

    return response