import langchain.chat_models
import langchain_core
import melo.api
import multiprocessing.shared_memory
import numpy as np
import os
import requests
import scipy.signal

modelMelo = melo.api.TTS('EN', 'cpu')
modelMelo.tts_to_file('Initialization', modelMelo.hps.data.spk2id['EN-Default'])

nSampleMax = 20 * 16000
memAudio = multiprocessing.shared_memory.SharedMemory('audio', create = True, size = np.dtype(np.float32).itemsize * nSampleMax)
ndarrAudio = np.ndarray([nSampleMax], dtype = np.float32, buffer = memAudio.buf)

if(not(os.environ.get('GOOGLE_API_KEY'))):
    # os.environ['GOOGLE_API_KEY'] = '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'
    os.environ['GOOGLE_API_KEY'] = 'AIzaSyA8HpwUgkWhSeCYPw9TjPjWjBTMEvd3q2o'

modelGemini = langchain.chat_models.init_chat_model('gemini-2.0-flash', model_provider = 'google_genai')
msgSys = langchain_core.messages.SystemMessage('Please answer only in paragraph format. Keep number of words in each sentence below 15.')

try:
    while(True):
        quest = input('You may ask a question:\n')
        ans = ''
        
        for token in modelGemini.stream([msgSys, langchain_core.messages.HumanMessage(quest)]):
            if('.' in token.content):
                listToken = token.content.split('.')
                ans += listToken[0] + '. '

                arrAudio = modelMelo.tts_to_file(ans, modelMelo.hps.data.spk2id['EN-Default'])
                nSample = round(arrAudio.shape[0] * 16000 / modelMelo.hps.data.sampling_rate)
                arrAudio = scipy.signal.resample(arrAudio, nSample)

                ndarrAudio[:arrAudio.shape[0]] = arrAudio
                requests.get(f'http://127.0.0.1:8000/{arrAudio.shape[0]}')

                if(len(listToken) > 0):
                    ans = '. '.join(listToken[1:])
                else:
                    ans = ''
            else:
                ans += token.content
finally:
    memAudio.unlink()
