from multiprocessing import Process
import subprocess
from flask import Flask, request, render_template
from chat_sampler import ChatSampler, parse_args
import os


class ChatServer(Process):
    def __init__(self):
        Process.__init__(self)
        my_args = parse_args()

        my_args.changes = "indx_word = './model/ivocab.in.pkl', \
                    indx_word_target = './model/ivocab.out.pkl', \
                    word_indx = './model/vocab.in.pkl', \
                    word_indx_trgt = './model/vocab.out.pkl'"
        self.sampler = ChatSampler(my_args)
        self.RESPONSE_MAX_COST = 1.5
        self.NUM_SAMPLES = 10
        self.tokenizer_cmd = [os.getcwd() + '/tokenizer.perl', '-l', 'en', '-q', '-']
        self.detokenizer_cmd = [os.getcwd() + '/detokenizer.perl', '-l', 'fr', '-q', '-']

    @staticmethod
    def reverse(input_chat):
        return ' '.join(input_chat.split(' ')[::-1])

    @staticmethod
    def filter_messages(messages):
        return [m for m in messages if 'UNK' not in m]

    def sample_chat(self, input_chat):
        seq = self.reverse(input_chat)
        return self.sampler.sample(seq, self.NUM_SAMPLES)

    def run(self):
        app = Flask(__name__)

        @app.route('/response_cost', methods=['GET', 'POST', 'PUT'])
        def response_cost_flask():
            if 'val' in request.args:
                self.RESPONSE_MAX_COST = float(request.args.get('val'))
            return 'Current max cost: %f' % self.RESPONSE_MAX_COST

        @app.route('/chat', methods=['GET', 'POST', 'PUT'])
        def chat_flask():
            debugging = 'debug' in request.args
            sentence = request.args.get('input')
            print 'Sentence: ', sentence
            tokenizer = subprocess.Popen(self.tokenizer_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            sentence, _ = tokenizer.communicate(sentence.encode('utf-8'))
            print 'Sentence after tokenize: ', sentence

            [messages, costs] = self.sample_chat(sentence)

            if costs is not None and len(costs) > 0:
                messages = self.filter_messages(messages)
                if len(messages) > 0:
                    for i in range(0, len(messages)):
                        detokenizer = subprocess.Popen(self.detokenizer_cmd, stdin=subprocess.PIPE,
                                                       stdout=subprocess.PIPE)
                        detokenized_sentence, _ = detokenizer.communicate(messages[i].encode('utf-8'))
                        messages[i] = detokenized_sentence
                    if debugging:
                        return '\n'.join(['%f\t%s' % (c, m) for (m, c) in zip(messages, costs)])
                    else:
                        return messages[0] if costs[0] < self.RESPONSE_MAX_COST else ""
            return ""

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
    # sample.py --beam-search --state your_state.pkl your_model.npz
    server = ChatServer()
    server.run()
