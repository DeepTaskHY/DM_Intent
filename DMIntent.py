#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import rospkg
from std_msgs.msg import String

import tensorflow as tf
import numpy as np
import os
import gc
import time
from model import Model
import pickle
from tqdm import tqdm
from sklearn.metrics import f1_score
from dataMaker import load_data, get_batches_norandom as get_batches, make_embedding_from_input
from google.protobuf.struct_pb2 import Struct
from google.protobuf import json_format

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/vision/YuriWorkspace/SocialRobot_IntentionClassification/authkey/socialrobot-hyu-xdtlug-eb5be21aa398.json"  # homecare
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/vision/YuriWorkspace/SocialRobot_IntentionClassification/authkey/socialrobot-hyu-reception-nyla-176b1cd7164b.json"  # reception
data_path = 'data/data_noHAN_kkma_190907_train.pkl'

'''
    Social Robot HYU
    DM (Intent Classification) model
    with Adaboost, ROS ver
'''


max_len, id2word, word2id, trainingSamples_list, validSamples, testSamples = load_data(data_path)

id2word = {}
for key in word2id.keys():
    id2word[word2id[key]] = key


def get_targets(samples):
    targets = []
    for sample in samples:
        targets.append(float(sample[1]))
    return np.array(targets)


# [START dialogflow_detect_intent_text]
def detect_intent_texts(project_id, session_id, texts, language_code):
    """Returns the result of detect intent with texts as inputs.
    Using the same `session_id` between requests allows continuation
    of the conversation."""

    import dialogflow_v2 as dialogflow
    session_client = dialogflow.SessionsClient()

    session = session_client.session_path(project_id, session_id)
    print('Session path: {}\n'.format(session))

    for text in texts:
        text_input = dialogflow.types.TextInput(
            text=text, language_code=language_code)

        query_input = dialogflow.types.QueryInput(text=text_input)

        response = session_client.detect_intent(
            session=session, query_input=query_input)

        print('=' * 20)
        print('Query text: {}'.format(response.query_result.query_text))
        # print('Detected intent: {} (confidence: {})\n'.format(
        #     response.query_result.intent.display_name,
        #     response.query_result.intent_detection_confidence))
        # print('Parameters text: {}\n'.format(response.query_result.parameters))
        # print('Parameters text type: \n', type(response.query_result.parameters))

        entities = response.query_result.parameters
        # print('entities type : ', type(entities))
        entities_dic = json_format.MessageToDict(entities)
        return entities_dic


# make response json
def make_response_json(label, human_speech, information):
    # print("##### information : ", information)
    label_list = {0: '정보전달', 1: '인사', 2: '질문', 3: '요청', 4: '약속', 5: '긍정', 6: '부정'}
    final_response = {
        "header": {
            "content": [
                "dialog_intent"
            ],
            "source": "dialog",
            "target": [
                "planning"
            ],
            "timestamp": str(time.time())
        },
        "dialog_intent": {
            "intent": label_list[label],
            "speech": human_speech,
            "name": "",
            "information": information
        }
    }
    return final_response


RNN_SIZE = 192
VOCABULARY_SIZE = len(word2id)
EMBEDDING_SIZE = 200
ATTENTION_SIZE = 50
LEARNING_RATE = 0.001
L2_LEG_LAMBDA = 3.0
EPOCH = 50
BATCH_SIZE = 64
N_LABEL = 7
DROPOUT_KEEP_PROB = 0.5
SEQ_LEN = max_len
INDEX_FRONT = word2id['<SPLIT>'] + 1
INDEX_BACK = VOCABULARY_SIZE - INDEX_FRONT
RNN_CELL = 'GRU'
MODEL = 'birnn'


def ros_callback_fn(msg):
    if msg.data != '':
        # convert ros message to json
        ros_input = json.loads(msg.data, encoding='utf-8')

        if "dialog" == ros_input['header']['target'][0]:
            print_for_check(" Input ", ros_input)

            name = ros_input['dialog_generation']['name']
            intent = ros_input['dialog_generation']['intent']
            human_speech = ros_input['dialog_generation']['human_speech']
            id = ros_input['dialog_generation']['id']
            social_context = ros_input['dialog_generation']['social_context']

            # input test
            data = human_speech

            AdaboostInfoLocation = 'data/AdaboostInfo_intent_ensemble_noJosaEomi_th5to2500_boost_30_181211_porting.pkl'
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            models = os.listdir('birnn_GRU_0.001_192_ENSEMBLE9/')
            #probabilities = np.array([[0., 0.] for i in range(len(testSamples))])
            probabilities = []
            # print('model', model, 'check....')
            # model_path = '/birnn_GRU_0.001_192_ENSEMBLE9/' + model + '/best_val'
            model = models.__getitem__(0)
            model_path = 'birnn_GRU_0.001_192_ENSEMBLE9/best_val'
            tf.reset_default_graph()
            with tf.Session(config=config) as sess:
                model = Model(rnn_size=RNN_SIZE, vocabulary_size=VOCABULARY_SIZE, sequence_len=SEQ_LEN, embedding_size=EMBEDDING_SIZE, attention_size=ATTENTION_SIZE, learning_rate=LEARNING_RATE, l2_reg_lambda=L2_LEG_LAMBDA, n_label=N_LABEL, index_front=INDEX_FRONT, index_back=INDEX_BACK, rnn_cell=RNN_CELL)
                ckpt = tf.train.get_checkpoint_state(model_path)
                if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                    model.saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    raise ValueError('No such file:[{}]'.format(model_path))

                idlist, sentence_length = make_embedding_from_input(data, word2id)

                # source = list(sample[0])
                pad = [0] * (SEQ_LEN - len(idlist))
                idlist = [idlist + pad]

                # with open('D:/model/result_rank.txt', 'w', encoding='UTF8') as f:

                feed_dict = {model.inputs: idlist,
                            model.inputs_length: [1],
                            model.targets: [0],
                            model.dropout_keep_prob: 1.0}

                idlist_probability = sess.run(tf.nn.softmax(model.logits), feed_dict=feed_dict)
                probabilities.append(idlist_probability)
                # print("### probabilities ", probabilities)
                # print("### idlist_probability ", idlist_probability)

                del model
                gc.collect()

            all_predictions = int(np.argmax(probabilities))

            # info = detect_intent_texts('socialrobot-hyu-xdtlug', 'hyusocialdmgenerator', [data], 'ko')  # homecare ko
            info = detect_intent_texts('socialrobot-hyu-reception-nyla', 'hyusocialdmgenerator', [data], 'ko')  # reception ko
            # print("intent: ", all_predictions, ' type: ', type(all_predictions))
            final = make_response_json(all_predictions, data, info)
            info = None

            # ROS
            task_completion_pub = rospy.Publisher('/dialog_intent', String, queue_size=10)
            task_completion_pub.publish(json.dumps(final, ensure_ascii=False, indent=4))

            print_for_check(" Output", final)
            print("="*100)


def run_subscriber():
	rospy.init_node('HYU_DM_Intention_Classification')
	rospy.Subscriber('/taskExecution', String, ros_callback_fn)
	rospy.spin()


if __name__ == '__main__':
    run_subscriber()
