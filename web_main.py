# coding:utf-8
'''
author:wangyi
'''
import tornado.web
import tornado.httpserver
import tornado.ioloop
import tornado.options
import numpy as np
import os
from tornado.options import define, options
import json
from predict import Predict

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('index.html')

class ResultHandler(tornado.web.RequestHandler):

    def post(self, *args, **kwargs):
        content = self.get_argument('content')
        res = predict.predict(content)
        self.write(json.dumps({'res':res},cls=MyEncoder))
        self.finish()



if __name__ == '__main__':

    predict = Predict('/home/abc/pySpace/disciplinary_action/datasets/stopwords.txt',
                      '/home/abc/pySpace/disciplinary_action/datasets/vocab.txt',
                      '/home/abc/pySpace/disciplinary_action/datasets/categories.txt',
                      '/home/abc/pySpace/disciplinary_action/checkpoints/best_validation')

    define("port", default=8599, help="run on the given port", type=int)
    tornado.options.parse_command_line()
    # 自定义settings
    settings = dict(
        template_path=os.path.join(os.path.dirname(__file__), "templates"),
        static_path=os.path.join(os.path.dirname(__file__), "statics"),  # 为了便于部署，建议使用static
        static_url_prefix="/statics/"  # 默认使用的是static，为了便于部署，建议使用static
    )
    app = tornado.web.Application(
        handlers=[(r'/', IndexHandler), (r'/predict', ResultHandler)], **settings, debug=True
    )
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()