# -*- coding: utf-8 -*-
import ctypes
import cv2
import numpy as np

import os, sys, io, gc, re
import glob, uuid, base64, json, time

from tornado.escape import json_encode

from tornado import websocket, web, ioloop

import tornado.ioloop
import tornado.web
import tornado.websocket


lib = ctypes.cdll.LoadLibrary('./lib.so')
lib.bar()
W = lib.getW()
H = lib.getH()
def imgbite():
           res = np.zeros(dtype=np.uint8, shape=(H, W, 4))
           lib.fdata(res.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))
           _, img_str = cv2.imencode('.jpg', res)
           BS = img_str.tobytes()
           return BS




class ImageWebSocket(tornado.websocket.WebSocketHandler):
    clients = set()

    def check_origin(self, origin):
        # Allow access from every origin
        return True

    def open(self):
        ImageWebSocket.clients.add(self)
        print("WebSocket opened from: " + self.request.remote_ip)

    def on_message(self, message):
        print (message)
        s = imgbite()
        self.write_message(s, binary=True)

    def on_close(self):
        ImageWebSocket.clients.remove(self)
        print("WebSocket closed from: " + self.request.remote_ip)

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html", title="Нейронная сеть/Тренировка")

app = tornado.web.Application([
        (r"/", MainHandler),
        (r"/websocket", ImageWebSocket),
    ])
app.listen("8800")


tornado.ioloop.IOLoop.current().start()


