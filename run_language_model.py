from transformers import AutoModelForCausalLM, AutoTokenizer
from just_playback import Playback
from tqdm.auto import trange
from TTS.api import TTS
import soundfile as sf
from PIL import Image
import librosa as lr
import numpy as np
import websockets
import subprocess
import threading
import tempfile
import pyaudio
import asyncio
import random
import queue
import torch
import time
import cv2
import re


# asyncio.run(main())
# print(TTS.list_models())
tts_model = TTS("tts_models/en/vctk/vits")


def say_tts(tts, text, callback=lambda vol: None, speaker_id="p300", sr=22050):
    # with tempfile.NamedTemporaryFile("wb", suffix=".wav") as tf:
        # temp_file = tf.name
        # subprocess.call(["tts", "--text", text, "--model_name", model_name, "--speaker_id", speaker_id, "--out_path", temp_file])
        # subprocess.call(["ffmpeg", "-y", "-i", temp_file, "-af", "asetrate=22050*4/3,atempo=3/4", out_file])
    wav = tts.tts(text, speaker=speaker_id)
    wav_ = lr.effects.pitch_shift(np.asarray(wav), sr, n_steps=6)
    # sf.write("result.wav", wav_, sr)
    
    vols = (np.lib.stride_tricks.sliding_window_view(wav_, (sr // 2,)) ** 2).mean(-1)
    vols /= vols.max()
    vols = np.sin(np.arange(len(vols)) / sr * 6) * (vols > 0.33)
    playback = Playback()
    with tempfile.NamedTemporaryFile("wb", suffix=".wav") as tf:
        sf.write(tf.name, wav_, sr)
        playback.load_file(tf.name)
    playback.play()
    while playback.active:
        time.sleep(0.01)
        callback(vols[min(int(playback.curr_pos * sr), len(vols) - 1)])

q = queue.Queue()

def run_poser(q):
    MODEL_NAME = "standard_float"
    device = torch.device("cuda")

    def load_poser(model: str, device: torch.device):
        print("Using the %s model." % model)
        if model == "standard_float":
            from tha3.poser.modes.standard_float import create_poser
            return create_poser(device)
        elif model == "standard_half":
            from tha3.poser.modes.standard_half import create_poser
            return create_poser(device)
        elif model == "separable_float":
            from tha3.poser.modes.separable_float import create_poser
            return create_poser(device)
        elif model == "separable_half":
            from tha3.poser.modes.separable_half import create_poser
            return create_poser(device)
        else:
            raise RuntimeError("Invalid model: '%s'" % model)
            
    poser = load_poser(MODEL_NAME, device.type)
    poser.get_modules()

    from tha3.poser.modes.pose_parameters import get_pose_parameters
    pose_parameters = get_pose_parameters()
    mouth_index = pose_parameters.get_parameter_index(f"mouth_aaa")
    # mouth_left_index = pose_parameters.get_parameter_index(f"mouth_aaa_left")
    # mouth_right_index = pose_parameters.get_parameter_index(f"mouth_aaa_right")
    iris_small_left_index = pose_parameters.get_parameter_index("iris_small_left")
    iris_small_left_index = pose_parameters.get_parameter_index("iris_small_left")
    iris_small_right_index = pose_parameters.get_parameter_index("iris_small_right")
    iris_rotation_x_index = pose_parameters.get_parameter_index("iris_rotation_x")
    iris_rotation_y_index = pose_parameters.get_parameter_index("iris_rotation_y")
    head_x_index = pose_parameters.get_parameter_index("head_x")
    head_y_index = pose_parameters.get_parameter_index("head_y")
    neck_z_index = pose_parameters.get_parameter_index("neck_z")
    body_y_index = pose_parameters.get_parameter_index("body_y")
    body_z_index = pose_parameters.get_parameter_index("body_z")
    breathing_index = pose_parameters.get_parameter_index("breathing")

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


    from tha3.util import resize_PIL_image, extract_PIL_image_from_filelike, \
        extract_pytorch_image_from_PIL_image, convert_output_image_from_torch_to_numpy
    pil_image = Image.open("/home/kqjacs/AFS/lm-benchmark-hook/data/images/crypko_00.png")
    torch_input_image = extract_pytorch_image_from_PIL_image(pil_image).to(device)
    pose_size = poser.get_num_parameters()
    pose = torch.zeros(1, pose_size, dtype=poser.get_dtype(), device=device)
    i = 0
    val = 0
    while True:
        try:
            val = q.get_nowait()
        except queue.Empty:
            pass
        pose[0, breathing_index] = np.sin(i / 20)
        pose[0, mouth_index] = val
        # pose[0, mouth_left_index] = val
        # pose[0, mouth_right_index] = val
        output_image = poser.pose(torch_input_image, pose)[0]
        # output_image = pytorch_image.detach().cpu()
        numpy_image = np.uint8(np.rint(convert_output_image_from_torch_to_numpy(output_image.detach().cpu()) * 255.0))
        pil_image = Image.fromarray(numpy_image, mode='RGBA')
        image = np.asarray(pil_image)
        green = image[..., :-1].copy()
        green[..., [0, 2]] = 0
        green[..., 1] = 255
        alpha = image[..., -1:] > 0
        image = image[..., :-1] * alpha + (~alpha) * green
        cv2.imshow("frame", image[..., ::-1])
        cv2.waitKey(1)
        # pil_image.save(f"outputs/anime{i:03d}.png")
        i += 1
        # time.sleep(0.01)


threading.Thread(target=run_poser, args=(q,)).start()
# time.sleep(4)
def putify(x):
    with q.mutex:
        q.queue.clear()
    q.put(x)
# say_tts(tts_model, "Hello world. I am a bot", callback=putify)



random.seed(2)
torch.set_grad_enabled(False)
text = open("prompt").read()
prompt, *qa = list(open("prompt"))
qs, a = qa[::2], qa[1::2]
qa = list(zip(qs, a))
model_name = "EleutherAI/pythia-2.8b-deduped"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                            low_cpu_mem_usage=True,
                                            load_in_8bit=True,
                                            device_map="auto"
                                            )


def answer(qo):
    random.shuffle(qa)
    text = prompt + "".join(qu + a for qu, a in qa[:10]) + f"Q: {qo}\n"
    tokens = torch.LongTensor(tokenizer.encode(text)).unsqueeze(0)
    result = model.generate(input_ids=tokens.to(model.device), max_new_tokens=300, temperature=0.96, repetition_penalty=1.0, length_penalty=0.25,
                            # penalty_alpha=0.6,
                            do_sample=True, eos_token_id=50, pad_token_id=tokenizer.pad_token_id,
    )
    ans = tokenizer.decode(result[0, tokens.shape[1] + 2:-1].detach().cpu().numpy().tolist())
    say_tts(tts_model, ans, callback=putify)
    return ans


TWITCH_NAME = "kqjacs"
_cmd_pat = (
    "^(@(?P<tags>[^ ]*) )?(:(?P<prefix>[^ ]+) +)?"
    "(?P<command>[^ ]+)( *(?P<argument> .+))?"
)
_rfc_1459_command_regexp = re.compile(_cmd_pat)


chat_queue = queue.Queue()
async def socket(chat_queue):
    async with websockets.connect("wss://irc-ws.chat.twitch.tv:443") as ws:
        await ws.send("CAP REQ :twitch.tv/membership twitch.tv/tags twitch.tv/commands")
        await ws.send("PASS " + open("twitch_oauth").read())
        await ws.send(f"NICK {TWITCH_NAME}")
        await ws.send(f"JOIN #{TWITCH_NAME}")
        while True:
            text = await ws.recv()
            grp = _rfc_1459_command_regexp.match(text).group
            cmd, arg = grp("command").strip(), grp("argument")
            if cmd == "PING":
                await ws.send("PONG")
            elif cmd == "PRIVMSG":
                chat_queue.put(arg.partition(":")[-1])


torch.manual_seed(12)
launch_thread = lambda chat_queue: asyncio.run(socket(chat_queue))
threading.Thread(target=launch_thread, args=(chat_queue,)).start()
print("all loaded")
async def main():
    while True:
        que = chat_queue.get()
        print("", "que", que)
        say_tts(tts_model, que[0], callback=putify)
        answer(que)
asyncio.run(main())