import os
from multiprocessing import Process
import subprocess
from flask import Flask, request, render_template, jsonify
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from chat_sampler import ChatSampler, parse_args
import conf


def get_what_was_that(language):
    if language == conf.LANGUAGES[0]:
        return 'What was that?'
    elif language == conf.LANGUAGES[1]:
        return 'Xin l\xe1\xbb\x97i t\xc3\xb4i kh\xc3\xb4ng hi\xe1\xbb\x83u :)'
    else:
        raise ValueError('Unknown language: {}'.format(language))


def reverse(input_chat):
    return ' '.join(input_chat.split(' ')[::-1])


def filter_messages(messages, costs):
    return [(m, c) for m, c in zip(messages, costs) if 'UNK' not in m and len(m.strip()) > 0]


def sample_chat(input_chat, sampler):
    seq = reverse(input_chat)
    return sampler.sample(seq, conf.NUM_SAMPLES)


class ChatServer(Process):
    def __init__(self):
        Process.__init__(self)
        self.samplers = {}
        for l in conf.LANGUAGES:
            my_args = parse_args()
            my_args = conf.get_config(l, my_args)
            print(my_args)
            self.samplers[l] = ChatSampler(my_args)

        self.tokenizer_cmd = [os.getcwd() + '/tokenizer.perl', '-l', 'en', '-q', '-']
        self.detokenizer_cmd = [os.getcwd() + '/detokenizer.perl', '-l', 'fr', '-q', '-']

    def run(self):
        app = Flask(__name__)

        app.config.from_object(conf)

        @app.route('/response_cost')
        def response_cost_flask():
            if 'val' in request.args:
                conf.RESPONSE_MAX_COST = float(request.args.get('val'))
            return 'Current max cost: %f' % conf.RESPONSE_MAX_COST

        @app.route('/chat', methods=['GET'])
        def chat_flask():
            try:
                debugging = 'debug' in request.args
                sentence = request.args.get('input')
                language = request.args.get('lang')

                if language not in self.samplers:
                    return jsonify(error=True, message='Invalid language: {}'.format(language))

                print 'Sentence: ', sentence
                tokenizer = subprocess.Popen(self.tokenizer_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                sentence, _ = tokenizer.communicate(sentence.encode('utf-8'))
                print 'Sentence after tokenize: ', sentence

                [messages, costs] = sample_chat(sentence, self.samplers[language])

                if costs is not None and len(costs) > 0:
                    messages = filter_messages(messages, costs)
                    detokenized_messages = []
                    for m, c in messages:
                        try:
                            detokenizer = subprocess.Popen(self.detokenizer_cmd, stdin=subprocess.PIPE,
                                                           stdout=subprocess.PIPE)
                            detokenized_sentence, _ = detokenizer.communicate(m)
                            detokenized_messages.append((detokenized_sentence, c))
                        except (UnicodeEncodeError, UnicodeDecodeError) as ex:
                            print 'Error detokenizing: ', m
                            print ex
                    if debugging:
                        return jsonify(debug=True, messages=[m for m, _ in detokenized_messages],
                                       costs=[c for _, c in detokenized_messages])
                    return jsonify(message=detokenized_messages[0][0] if
                                   detokenized_messages[0][1] < conf.RESPONSE_MAX_COST else
                                   get_what_was_that(language))
                return jsonify(message=get_what_was_that(language))

            except (UnicodeEncodeError, UnicodeDecodeError) as ex:
                return jsonify(error=True, message='{}'.format(ex))

        @app.route('/shutdown', methods=['GET', 'POST', 'PUT'])
        def terminate_flask():
            func = request.environ.get('werkzeug.server.shutdown')
            if func is None:
                raise RuntimeError('Not running with the Werkzeug Server')
            func()
            return 'OK'

        @app.route('/')
        def index():
            return render_template('index.html')

        # app.run(host='0.0.0.0', port=conf.SERVICE_PORT, threaded=True, use_reloader=False)

        http_server = HTTPServer(WSGIContainer(app))
        http_server.listen(conf.SERVICE_PORT)
        IOLoop.instance().start()


if __name__ == '__main__':
    server = ChatServer()
    server.run()
