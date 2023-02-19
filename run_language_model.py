from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import trange
import asyncio
import random
import torch


# async def main():
    # await asyncio.sleep(1)
    # print('hello')


# asyncio.run(main())


random.seed(2)
torch.set_grad_enabled(False)
text = open("prompt").read()
prompt, *qa = list(open("prompt"))
q, a = qa[::2], qa[1::2]
qa = list(zip(q, a))
model_name = "EleutherAI/pythia-2.8b-deduped"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             low_cpu_mem_usage=True,
                                             load_in_8bit=True,
                                             device_map="auto"
                                             )
# tts --text "Hello world! I am saying something. After the person broke the rode, the ffffffffff author didn't >_<." --model_name "tts_models/en/vctk/vits" --speaker_id "p300" --out_path output.wav
# ffmpeg -y -i output.wav -af 'asetrate=22050*4/3,atempo=3/4'  pitchshift.wav

# import torch
# MODEL_NAME = "standard_float"
# DEVICE_NAME = 'cuda'
# device = torch.device(DEVICE_NAME)

# def load_poser(model: str, device: torch.device):
#     print("Using the %s model." % model)
#     if model == "standard_float":
#         from tha3.poser.modes.standard_float import create_poser
#         return create_poser(device)
#     elif model == "standard_half":
#         from tha3.poser.modes.standard_half import create_poser
#         return create_poser(device)
#     elif model == "separable_float":
#         from tha3.poser.modes.separable_float import create_poser
#         return create_poser(device)
#     elif model == "separable_half":
#         from tha3.poser.modes.separable_half import create_poser
#         return create_poser(device)
#     else:
#         raise RuntimeError("Invalid model: '%s'" % model)
        
# poser = load_poser(MODEL_NAME, DEVICE_NAME)
# poser.get_modules()



# iris_small_left_index = pose_parameters.get_parameter_index("iris_small_left")
# iris_small_right_index = pose_parameters.get_parameter_index("iris_small_right")
# iris_rotation_x_index = pose_parameters.get_parameter_index("iris_rotation_x")
# iris_rotation_y_index = pose_parameters.get_parameter_index("iris_rotation_y")
# head_x_index = pose_parameters.get_parameter_index("head_x")
# head_y_index = pose_parameters.get_parameter_index("head_y")
# neck_z_index = pose_parameters.get_parameter_index("neck_z")
# body_y_index = pose_parameters.get_parameter_index("body_y")
# body_z_index = pose_parameters.get_parameter_index("body_z")
# breathing_index = pose_parameters.get_parameter_index("breathing")

# def get_pose():
#     pose = torch.zeros(1, pose_size, dtype=poser.get_dtype())

#     eyebrow_name = f"eyebrow_{eyebrow_dropdown.value}"
#     eyebrow_left_index = pose_parameters.get_parameter_index(f"{eyebrow_name}_left")
#     eyebrow_right_index = pose_parameters.get_parameter_index(f"{eyebrow_name}_right")
#     pose[0, eyebrow_left_index] = eyebrow_left_slider.value
#     pose[0, eyebrow_right_index] = eyebrow_right_slider.value

#     eye_name = f"eye_{eye_dropdown.value}"
#     eye_left_index = pose_parameters.get_parameter_index(f"{eye_name}_left")
#     eye_right_index = pose_parameters.get_parameter_index(f"{eye_name}_right")
#     pose[0, eye_left_index] = eye_left_slider.value
#     pose[0, eye_right_index] = eye_right_slider.value

#     mouth_name = f"mouth_{mouth_dropdown.value}"
#     if mouth_name == "mouth_lowered_corner" or mouth_name == "mouth_raised_corner":
#         mouth_left_index = pose_parameters.get_parameter_index(f"{mouth_name}_left")
#         mouth_right_index = pose_parameters.get_parameter_index(f"{mouth_name}_right")
#         pose[0, mouth_left_index] = mouth_left_slider.value
#         pose[0, mouth_right_index] = mouth_right_slider.value
#     else:
#         mouth_index = pose_parameters.get_parameter_index(mouth_name)
#         pose[0, mouth_index] = mouth_left_slider.value

#     pose[0, iris_small_left_index] = iris_small_left_slider.value
#     pose[0, iris_small_right_index] = iris_small_right_slider.value
#     pose[0, iris_rotation_x_index] = iris_rotation_x_slider.value
#     pose[0, iris_rotation_y_index] = iris_rotation_y_slider.value
#     pose[0, head_x_index] = head_x_slider.value
#     pose[0, head_y_index] = head_y_slider.value
#     pose[0, neck_z_index] = neck_z_slider.value
#     pose[0, body_y_index] = body_y_slider.value
#     pose[0, body_z_index] = body_z_slider.value
#     pose[0, breathing_index] = breathing_slider.value

#     return pose.to(device)

    # output_image = pytorch_image.detach().cpu()
    # numpy_image = numpy.uint8(numpy.rint(convert_output_image_from_torch_to_numpy(output_image) * 255.0))
    # pil_image = PIL.Image.fromarray(numpy_image, mode='RGBA')

    # output_image = poser.pose(torch_input_image, pose)[0]

samples = 16
for i in range(samples):
    random.shuffle(qa)
    text = prompt + "".join(q + a for q, a in qa[:10]) + "Q: What is your real name?\n"
    tokens = torch.LongTensor(tokenizer.encode(text)).unsqueeze(0)
    torch.manual_seed(12)
    result = model.generate(input_ids=tokens.to(model.device), max_new_tokens=100, temperature=0.96, repetition_penalty=1.0, length_penalty=0.25,
                            # penalty_alpha=0.6,
                            do_sample=True, eos_token_id=50, pad_token_id=tokenizer.pad_token_id, max_new_length=200
    )
    answer = tokenizer.decode(result[0, tokens.shape[1] + 2:-1].detach().cpu().numpy().tolist())
    print(answer)