#!/bin/sh

THEANO_FLAGS="floatX=float32" python chat_server.py --beam-search --state model/encdec_state.pkl model/encdec_model.npz