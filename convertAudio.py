import contextlib
import fastapi
import multiprocessing
import multiprocessing.shared_memory
import numpy as np
import os
import requests
import scipy.signal
import threading
import uvicorn

import configs.config
import infer.modules.vc.modules
import musetalk.utils.audio_processor

pathRvc = os.path.join('/home', 'yhyeo0202', 'talkHead', 'Retrieval-based-Voice-Conversion-WebUI')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['no_proxy'] = 'localhost, 127.0.0.1, ::1'
os.environ['weight_root'] = os.path.join(pathRvc, 'assets', 'weights')
os.environ['index_root'] = os.path.join(pathRvc, 'logs')
os.environ['rmvpe_root'] = os.path.join(pathRvc, 'assets', 'rmvpe')

vc = infer.modules.vc.modules.VC(configs.config.Config())
vc.get_vc('kurz.pth', 0.33, 0.33)
pathIndex = os.path.join(pathRvc, 'logs', 'kurzgesagt', 'trained_IVF1574_Flat_nprobe_1_kurzgesagt_v2.index')
vc.vcInMem(0, np.zeros([16000], np.float32), 0.0, None, 'rmvpe', pathIndex, '', 0.75, 3, 0, 0.25, 0.33)

nSampleMax = 20 * 16000
memAudio = multiprocessing.shared_memory.SharedMemory('audio')
ndarrAudio = np.ndarray([nSampleMax], dtype = np.float32, buffer = memAudio.buf)

memAudioConvert = multiprocessing.shared_memory.SharedMemory('audioConvert', create = True, size = np.dtype(np.float32).itemsize * nSampleMax)
ndarrAudioConvert = np.ndarray([nSampleMax], dtype = np.float32, buffer = memAudioConvert.buf)

lock = threading.Lock()
app = fastapi.FastAPI()

@app.get("/{nSample}")
async def root(nSample : int):
    lock.acquire()

    arrAudio = vc.vcInMem(0, ndarrAudio[:nSample], 0.0, None, 'rmvpe', pathIndex, '', 0.75, 3, 0, 0.25, 0.33)
    nSample = round(arrAudio.shape[0] * 16000 / 40000)
    arrAudio = scipy.signal.resample(arrAudio, nSample)
    
    ndarrAudioConvert[:arrAudio.shape[0]] = arrAudio
    requests.get(f'http://127.0.0.2:8000/{arrAudio.shape[0]}')

    lock.release()

    return

uvicorn.run(app, host = '127.0.0.1', port = 8000)

memAudio.close()
multiprocessing.resource_tracker.unregister(memAudio._name, 'shared_memory')
memAudioConvert.unlink()
