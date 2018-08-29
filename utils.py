import keras.backend as K
import functools
import tensorflow as tf

def load_dtypes(dtypes_path):
    dtypes = {}
    with open(dtypes_path) as fp:
        for row in fp.readlines():
            row_s = row.strip().split(",")
            name = row_s[0]
            dtyp = row_s[1]
            if name.startswith("hr_") and name != "hr_OrderOfFinish":
                continue
            dtypes[name] = dtyp
    return dtypes

def load_removes(remove_path):
    ls = []
    with open(remove_path) as fp:
        for row in fp.readlines():
            ls.append(row.strip())
    return sorted(ls)

def load_targets(remove_path):
    ls = []
    with open(remove_path) as fp:
        for row in fp.readlines():
            ls.append(row.strip())
    return sorted(ls)

def cats_and_nums(usecols,dtypes):
    cats = []
    nums = []
    for k,v in dtypes.items():
        if v == "object":
            cats.append(k)
        else:
            nums.append(k)
    cats = sorted(cats)
    nums = sorted(nums)
    return (cats,nums)

def as_keras_metric(method):
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

def label_encoder(df):
    pass


