from ChatTTS import ChatTTS
from IPython.display import Audio
import scipy
import torch

class TTS:
    def __init__(self):
        self.chat = ChatTTS.Chat() # OOP :)
        self.chat.load()

    def get_rand_spk(self):
        self.spk = self.chat.sample_random_speaker() # generate a tensor representing the voice of a random speaker

    def text_to_speech(self, texts):
        self.get_rand_spk() # tensor saved as self.spk instance variable

        params_infer_code = ChatTTS.Chat.InferCodeParams( # specify parameters for speech generation
            spk_emb = self.spk, # uses the random speaker tensor to voice the speech
                                # spk_emb = torch.load("tensor.pt") to load the previously saved speaker
            temperature = 0.3, # default temperature
            top_P = 0.7, # default top P
            top_K = 20 # default top K
        )

        # generate speech using the above specified parameters
        wavs = self.chat.infer(texts, skip_refine_text=True, params_infer_code=params_infer_code)

        # save generated audio as wav file
        Audio(wavs[0], rate=24_000, autoplay=True)
        wav_file_name = "output.wav" # specify file name (.wav)
        scipy.io.wavfile.write(filename=wav_file_name, rate=24_000, data=wavs[0].T)
        print(f"Output saved to {wav_file_name}. Verify output quality before continuing")

        self.save_spk() # initiate save speaker tensor subprocess
    
    def save_spk(self):
        if input("Save current speaker? (y/n) ") == "y":
            pt_file_name = "tensor.pt" # specify file name (.pt)
            torch.save(self.spk, pt_file_name)  # save the speaker tensor used to generate the audio
            print(f"Speaker saved to {pt_file_name}")
        else:
            print("Process terminated")
        
tts = TTS() # create TTS object
texts = ["Hello, World!"] # input text
tts.text_to_speech(texts)