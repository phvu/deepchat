import os
from multiprocessing import Process
import subprocess
from flask import Flask, request, render_template, jsonify
from chat_sampler import ChatSampler, parse_args
import conf


def get_config(language):
    my_args = parse_args()
    my_args.beam_search = True
    if language == conf.LANGUAGES[0]:
        my_args.model_path = 'model/encdec_model.npz'
        my_args.state = 'model/encdec_state.pkl'
        my_args.changes = "indx_word = './model/ivocab.in.pkl', " \
                          "indx_word_target = './model/ivocab.out.pkl', " \
                          "word_indx = './model/vocab.in.pkl', " \
                          "word_indx_trgt = './model/vocab.out.pkl'"
    elif language == conf.LANGUAGES[1]:
        my_args.model_path = 'model/encdec_model.npz'
        my_args.state = 'model/encdec_state.pkl'
        my_args.changes = "indx_word = './model/ivocab.in.pkl', " \
                          "indx_word_target = './model/ivocab.out.pkl', " \
                          "word_indx = './model/vocab.in.pkl', " \
                          "word_indx_trgt = './model/vocab.out.pkl'"
    else:
        raise ValueError('Unknown language: {}'.format(language))
    return my_args


def get_what_was_that(language):
    if language == conf.LANGUAGES[0]:
        return 'What was that?'
    elif language == conf.LANGUAGES[1]:
        return 'Xin loi toi khong hieu.'
    else:
        raise ValueError('Unknown language: {}'.format(language))


def reverse(input_chat):
    return ' '.join(input_chat.split(' ')[::-1])


def filter_messages(messages):
    return [m for m in messages if 'UNK' not in m and len(m.strip()) > 0]


def sample_chat(input_chat, sampler):
    seq = reverse(input_chat)
    return sampler.sample(seq, conf.NUM_SAMPLES)


class ChatServer(Process):
    def __init__(self):
        Process.__init__(self)
        self.samplers = {}
        for l in conf.LANGUAGES:
            my_args = get_config(l)
            print(my_args)
            self.samplers[l] = ChatSampler(my_args)

        self.tokenizer_cmd = [os.getcwd() + '/tokenizer.perl', '-l', 'en', '-q', '-']
        self.detokenizer_cmd = [os.getcwd() + '/detokenizer.perl', '-l', 'fr', '-q', '-']

    def run(self):
        app = Flask(__name__)

        @app.route('/response_cost')
        def response_cost_flask():
            if 'val' in request.args:
                conf.RESPONSE_MAX_COST = float(request.args.get('val'))
            return 'Current max cost: %f' % conf.RESPONSE_MAX_COST

        @app.route('/chat', methods=['GET'])
        def chat_flask():

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
                messages = filter_messages(messages)
                if len(messages) > 0:
                    for i in range(0, len(messages)):
                        detokenizer = subprocess.Popen(self.detokenizer_cmd, stdin=subprocess.PIPE,
                                                       stdout=subprocess.PIPE)
                        detokenized_sentence, _ = detokenizer.communicate(messages[i].encode('utf-8'))
                        messages[i] = detokenized_sentence
                    if debugging:
                        return jsonify(debug=True, messages=messages, costs=costs)
                    return jsonify(
                        message=messages[0] if costs[0] < conf.RESPONSE_MAX_COST else get_what_was_that(language))
            return jsonify(message=get_what_was_that(language))

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

        print 'Listening to 0.0.0.0:5000...'
        app.run(host='0.0.0.0', debug=True, threaded=True, use_reloader=False)


if __name__ == '__main__':
    server = ChatServer()
    server.run()
