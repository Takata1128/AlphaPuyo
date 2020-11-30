import tensorflow as tf
from keras.models import load_model

input_keras_model = './resources/best.h5'
export_dir = './resources/best_pb'

if __name__ == '__main__':
    old_session = tf.keras.backend.get_session()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf.keras.backend.set_session(sess)
    model = load_model(input_keras_model)
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    signature = tf.saved_model.predict_signature_def(
        inputs={
            t.name: t for t in model.inputs}, outputs={
            t.name: t for t in model.outputs})
    builder.add_meta_graph_and_variables(
        sess,
        tags=[
            tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict': signature})
    builder.save(as_text=True)
    sess.close()
    tf.keras.backend.set_session(old_session)

    print('input_node_names:')
    for t in model.inputs:
        print(t.name)

    print('output_node_names:')
    for t in model.outputs:
        print(t.name)
