from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Attention, Concatenate

#파라미터 설정(테스트용)
embedding_dim = 256 # 임베딩 차원
units = 1024 # LSTM의 hidden unit 개수
target_vocab_size = 2000 # target 어휘 사전의 크기
input_vocab_size = 2000 # input 어휘 사전의 크기

def build_model(input_vocab_size, target_vocab_size, embedding_dim, units):
    # 인코더
    encoder_inputs = Input(shape=(None,))
    encoder_embedding = Embedding(input_vocab_size, embedding_dim)(encoder_inputs)
    encoder_lstm = LSTM(units, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    # 디코더
    decoder_inputs = Input(shape=(None,))
    decoder_embedding = Embedding(target_vocab_size, embedding_dim)
    decoder_embedding_outputs = decoder_embedding(decoder_inputs)
    decoder_lstm = LSTM(units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding_outputs, initial_state=encoder_states)

    # 어텐션 적용
    attention_layer = Attention()
    attention_outputs = attention_layer([decoder_outputs, encoder_outputs])

    # 어텐션의 결과와 디코더의 출력을 연결 
    decoder_concat_input = Concatenate(axis=-1)([decoder_outputs, attention_outputs])

    # Dense층
    decoder_dense = Dense(target_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_concat_input)

    # Model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model