import contextlib
import copy
import cv2
import fastapi
import glob
import multiprocessing
import multiprocessing.shared_memory
import numpy as np
import os
import pickle
import pycuda.driver
import pycuda.autoinit
import sounddevice
import tensorrt as trt
import time
import torch
import threading
import uvicorn

import musetalk.utils.audio_processor
import musetalk.utils.blending

class ModelTrt:
    def __init__(self, pathEngine, listIndSort = None):
        self.listIndSort = listIndSort
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        trt.init_libnvinfer_plugins(self.logger, namespace = "")

        with open(pathEngine, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.contextEngine = self.engine.create_execution_context()
        self.stream = pycuda.driver.Stream()
        self.listAddrMemDev = []

        self.listShapeIn = []
        self.listMemHostIn = []
        self.listMemDevIn = []

        self.listShapeOut = []
        self.listMemHostOut = []
        self.listMemDevOut = []

        for i in range(self.engine.num_io_tensors):
            nameTsr = self.engine.get_tensor_name(i)

            if(self.engine.get_tensor_mode(nameTsr) == trt.TensorIOMode.INPUT):
                self.listShapeIn.append(self.engine.get_tensor_shape(nameTsr))
                self.listMemHostIn.append(pycuda.driver.pagelocked_empty(trt.volume(self.listShapeIn[-1]), trt.nptype(self.engine.get_tensor_dtype(nameTsr))))
                self.listMemDevIn.append(pycuda.driver.mem_alloc(self.listMemHostIn[-1].nbytes))
                self.listAddrMemDev.append(int(self.listMemDevIn[-1]))
            else:
                self.listShapeOut.append(self.engine.get_tensor_shape(nameTsr))
                self.listMemHostOut.append(pycuda.driver.pagelocked_empty(trt.volume(self.listShapeOut[-1]), trt.nptype(self.engine.get_tensor_dtype(nameTsr))))
                self.listMemDevOut.append(pycuda.driver.mem_alloc(self.listMemHostOut[-1].nbytes))
                self.listAddrMemDev.append(int(self.listMemDevOut[-1]))

        return

    def __call__(self, listIn):
        for i in range(len(listIn)):
            np.copyto(self.listMemHostIn[i], listIn[i].ravel())
            pycuda.driver.memcpy_htod_async(self.listMemDevIn[i], self.listMemHostIn[i], self.stream)

        for i in range(self.engine.num_io_tensors):
            self.contextEngine.set_tensor_address(self.engine.get_tensor_name(i), self.listAddrMemDev[i])

        self.contextEngine.execute_async_v3(stream_handle = self.stream.handle)

        for i in range(len(self.listMemHostOut)):
            pycuda.driver.memcpy_dtoh_async(self.listMemHostOut[i], self.listMemDevOut[i], self.stream)

        self.stream.synchronize()
        listOut = []

        for i in range(len(self.listMemHostOut)):
            listOut.append(self.listMemHostOut[i].reshape(self.listShapeOut[i]))

        if(self.listIndSort is not None):
            listOut = [listOut[i] for i in self.listIndSort]

        return listOut

class UnetVaeDecode:
    def __init__(self):
        self.modelUnetEncode = ModelTrt(os.path.join('/home', 'yhyeo0202', 'talkHead', 'model', 'unetEncodeQuantIm', 'unetEncodeQuantIm.engine'), [13, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0])
        self.modelUnetDecode = ModelTrt(os.path.join('/home', 'yhyeo0202', 'talkHead', 'model', 'unetDecodeQuantIm', 'unetDecodeQuantIm.engine'))
        self.modelVaeDecode = ModelTrt(os.path.join('/home', 'yhyeo0202', 'talkHead', 'tmp', 'vaeDecodeQuant.engine'))

        return

    def __call__(self, tsrLatent, timeStep, tsrAudio):
        listOut = self.modelUnetEncode([tsrLatent.to('cpu').detach().numpy(), timeStep.to('cpu').detach().numpy(), tsrAudio.to('cpu').detach().numpy()])
        listOut = self.modelUnetDecode(listOut + [tsrAudio.to('cpu').detach()])
        listOut = self.modelVaeDecode(listOut)

        tsrOut = torch.from_numpy(listOut[0]).to('cuda')

        return tsrOut

def getFrame(listPath):
    listFrame = []

    for path in listPath:
        listFrame.append(cv2.imread(path))

    return listFrame

def latent2Frame(queueAudio, queueLatentWhisper, queueFrame):
    pathPreproc = os.path.join('/home', 'yhyeo0202', 'talkHead', 'MuseTalk', 'results', 'v15', 'avatars', 'avator_1')
    listLatent = torch.load(os.path.join(pathPreproc, 'latents.pt'))

    with open(os.path.join(pathPreproc, 'coords.pkl'), 'rb') as f:
        listCoo = pickle.load(f)

    listPathFrame = glob.glob(os.path.join(pathPreproc, 'full_imgs', '*.png'))
    listPathFrame = sorted(listPathFrame, key = lambda x : int(os.path.splitext(os.path.basename(x))[0]))
    listFrame = getFrame(listPathFrame)

    with open(os.path.join(pathPreproc, 'mask_coords.pkl'), 'rb') as f:
        listCooMask = pickle.load(f)

    listPathFrameMask = glob.glob(os.path.join(pathPreproc, 'mask', '*.png'))
    listPathFrameMask = sorted(listPathFrameMask, key = lambda x : int(os.path.splitext(os.path.basename(x))[0]))
    listFrameMask = getFrame(listPathFrameMask)

    indFrame = 0
    arrIndFrame = np.arange(len(listLatent))
    arrIndFrame = np.concatenate([arrIndFrame, arrIndFrame[1:-1][::-1]])

    modelUnetVaeDecode = UnetVaeDecode()
    modelUnetVaeDecode(torch.zeros([1, 8, 32, 32], dtype = torch.float32), torch.zeros([1], dtype = torch.int32), torch.zeros([1, 50, 384], dtype = torch.float32))

    while(True):
        arrAudio = queueAudio.get()
        tsrLatentWhisper = queueLatentWhisper.get()
        bPlayAudio = True

        for matLatentWhisper in tsrLatentWhisper:
            ind = arrIndFrame[indFrame]
            frameRecon = modelUnetVaeDecode(listLatent[ind], torch.tensor([0]), matLatentWhisper[None, ...])

            frameRecon = (frameRecon / 2 + 0.5).clamp(0, 1)
            frameRecon = frameRecon.detach().cpu().permute(0, 2, 3, 1).float().numpy()
            frameRecon = (frameRecon * 255).round().astype("uint8")
            frameRecon = frameRecon[..., ::-1][0, ...]

            colStart, rowStart, colEnd, rowEnd = listCoo[ind]
            frameRecon = cv2.resize(frameRecon, [colEnd - colStart, rowEnd - rowStart])
            frameInit = copy.deepcopy(listFrame[ind])

            frameBlend = musetalk.utils.blending.get_image_blending(frameInit, frameRecon, listCoo[ind], listFrameMask[ind], listCooMask[ind])
            queueFrame.put(frameBlend)

            if(bPlayAudio):
                sounddevice.play(arrAudio, 16000)
                bPlayAudio = False

            if(indFrame == (arrIndFrame.shape[0] - 1)):
                indFrame = 0
            else:
                indFrame += 1

    return

def frame2Video(queue):
    period = 0.1 - 0.005

    while(True):
        timeStart = time.time()
        frame = queue.get()

        cv2.imshow('debug', frame)

        while((time.time() - timeStart) < period):
            cv2.waitKey(1)

    return

if(__name__ == '__main__'):
    multiprocessing.set_start_method('spawn')
    memAudio = multiprocessing.shared_memory.SharedMemory('audioConvert')
    ndarrAudio = np.ndarray([20 * 16000], dtype = np.float32, buffer = memAudio.buf)

    procAudio = musetalk.utils.audio_processor.AudioProcessor(feature_extractor_path = os.path.join('/home', 'yhyeo0202', 'talkHead', 'MuseTalk', 'models', 'whisper'))
    modelWhisper = ModelTrt(os.path.join('/home', 'yhyeo0202', 'talkHead', 'tmp', 'whisperEncodeSearchQuant.engine'))
    listFeatAudio = procAudio.getFeatAudio(np.zeros([16000], np.float32))
    procAudio.getLatentWhisper(listFeatAudio, torch.float32, modelWhisper, 16000, 12)

    queueFrame = multiprocessing.Queue()
    procFrame2Video = multiprocessing.Process(target = frame2Video, args = (queueFrame,))
    procFrame2Video.start()

    queueAudio = multiprocessing.Queue()
    queueLatentWhisper = multiprocessing.Queue()
    procLatent2Frame = multiprocessing.Process(target = latent2Frame, args = (queueAudio, queueLatentWhisper, queueFrame))
    procLatent2Frame.start()

    lock = threading.Lock()
    app = fastapi.FastAPI()
    
    @app.get("/{nSample}")
    async def root(nSample : int):
        lock.acquire()

        listFeatAudio = procAudio.getFeatAudio(ndarrAudio[:nSample])
        tsrLatentWhisper = procAudio.getLatentWhisper(listFeatAudio, torch.float32, modelWhisper, nSample, 10)

        queueAudio.put(ndarrAudio[:nSample])
        queueLatentWhisper.put(tsrLatentWhisper)

        lock.release()

        return
    
    uvicorn.run(app, host = '127.0.0.2', port = 8000)

    memAudio.close()
    multiprocessing.resource_tracker.unregister(memAudio._name, 'shared_memory')
