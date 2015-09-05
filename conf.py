import os

# flask
SECRET_KEY = '\xbf\xfc&1T\x90\xe6\x8c\x8d\xd3\x91\x08\xaaRJ\x9c\x8aaw\xa5\x17\xe3\xbd\xa9'
DEBUG = True

SERVICE_PORT = 5000

RESPONSE_MAX_COST = 10
NUM_SAMPLES = 5
LANGUAGES = ['english', 'vietnamese']


def get_config(language, my_args):
    if language not in LANGUAGES:
        raise ValueError('Unknown language: {}'.format(language))

    my_args.beam_search = True
    if language == LANGUAGES[0]:
        path_prefix = os.path.join(os.path.split(__file__)[0], 'model/english/')
        my_args.model_path = os.path.join(path_prefix, 'subscene_english_model.npz')
        my_args.state = os.path.join(path_prefix, 'subscene_english_state.pkl')
        my_args.changes = "indx_word = '{p}/ivocab.in.pkl', " \
                          "indx_word_target = '{p}/ivocab.out.pkl', " \
                          "word_indx = '{p}/vocab.in.pkl', " \
                          "word_indx_trgt = '{p}/vocab.out.pkl'".format(p=path_prefix)
    elif language == LANGUAGES[1]:
        path_prefix = os.path.join(os.path.split(__file__)[0], 'model/vietnamese/')
        my_args.model_path = os.path.join(path_prefix, 'subscene_vietnamese_model.npz')
        my_args.state = os.path.join(path_prefix, 'subscene_vietnamese_state.pkl')
        my_args.changes = "indx_word = '{p}/ivocab.in.pkl', " \
                          "indx_word_target = '{p}/ivocab.out.pkl', " \
                          "word_indx = '{p}/vocab.in.pkl', " \
                          "word_indx_trgt = '{p}/vocab.out.pkl'".format(p=path_prefix)
    return my_args
