import numpy
from keras.models import clone_model
from keras import layers
from keras.models import Model
from keras.models import model_from_json
import math
import copy
import time
def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

class TextCNNExplainer :
    def __init__(self, tokenizer,class_names=None) :
        self.tokenizer = tokenizer
        self.class_names = class_names

    def predict(self, model, data):
        out = model.predict([data]*len(self.kernel_sizes))

    def compute_contrib_penultimate_layer(self, model, data, rule='L2'):
        n_chanels = len([layer for layer in model.layers if isinstance(layer, layers.InputLayer)])
        penultimate_layer = model.layers[-2]
        output_layer = model.layers[-1]
        ow = output_layer.get_weights()[0]
        out = model.predict([data] * n_chanels)
        #ow = model.get_layer("dense_2").get_weights()[0]
        intermediate_model = Model(inputs=model.input, outputs=penultimate_layer.output)
        intermediate_out = intermediate_model.predict([data]*n_chanels)
        i = 0
        contribs = numpy.empty((intermediate_out.shape[0],ow.shape[0], ow.shape[1]), dtype=numpy.float32)
        for out,c in zip(intermediate_out,out):
            out = out.reshape((out.shape[0],1))
            contrib = out * ow
            if rule == 'L2' : # Apply norm L2
                z = numpy.linalg.norm(contrib, ord=2, axis=0)
                contrib = contrib / z
            elif rule == 'L1' : # Apply L1-Norm
                z = numpy.linalg.norm(contrib, ord=1, axis=0)
                contrib = contrib / z
            elif rule == 'LRP-0' :
            # standard LRP -- uncomment the line below
                z = numpy.sum(contrib, axis=0)
                contrib = contrib / z
            elif rule == 'ABS' :
                z = numpy.sum(contrib, axis=0)
                contrib = contrib / abs(z)
            elif rule == 'INF':  # apply norm Inf
                z = numpy.linalg.norm(contrib, ord=math.inf, axis=0)
                contrib = contrib / z
            elif rule == 'PN' :
                contrib_pos = numpy.where(contrib>=0,contrib,0) # positive contributions
                contrib_pos = contrib_pos/(contrib_pos.sum(0)+0.00000001) # positive contributions percentage
                contrib_neg = numpy.where(contrib<0,contrib,0)
                contrib_neg = contrib_neg / (contrib_neg.sum(0)+0.0000001)
                contrib = contrib_neg + contrib_pos
            contrib = contrib*c
            contribs[i] = contrib

            i += 1
        return contribs

    def compute_contrib_pooling_layer(self, model, data, rule='L2'):
        #next(x for x in model.layers[::-1] if isinstance(x, layers.Conv2D))
        i = -3
        n_chanels = len([layer for layer in model.layers if isinstance(layer, layers.InputLayer)])
        if isinstance(model.layers[i+1], layers.Dense) :
            current_layer_contribs = None
            next_layer_contribs = self.compute_contrib_penultimate_layer(model, data, rule)
            while isinstance(model.layers[i+1], layers.Dense) :
                x = model.layers[i]
                next_layer = model.layers[i+1]
                weights = next_layer.get_weights()[0]
                intermediate_model = Model(inputs=model.input, outputs=x.output)
                intermediate_out = intermediate_model.predict([data] * n_chanels)
                j = 0
                print(x.name, next_layer.name)
                current_layer_contribs = numpy.empty((intermediate_out.shape[0], weights.shape[0], next_layer_contribs.shape[2]), dtype=numpy.float32)
                for (out, c) in zip(intermediate_out, next_layer_contribs):
                    out_1 = out.reshape((out.shape[0], 1))
                    contrib_mat = out_1 * weights
                    if rule == 'L2':  # Apply norm
                        z = numpy.linalg.norm(contrib_mat, ord=2, axis=0)
                        contrib_mat = contrib_mat / z
                        #contrib_mat = numpy.where(contrib_mat==math.inf or contrib_mat == math.nan, 0, contrib_mat)
                    elif rule == 'L1':  # Apply L1-Norm
                        z = numpy.linalg.norm(contrib_mat, ord=1, axis=0)
                        contrib_mat = contrib_mat / z
                    elif rule == 'INF' : # apply norm Inf
                        z = numpy.linalg.norm(contrib_mat, ord=math.inf, axis=0)
                        contrib_mat = contrib_mat / z
                    elif rule == 'LRP-0':
                        # standard LRP -- uncomment the line below
                        z = numpy.sum(contrib_mat, axis=0)
                        contrib_mat = contrib_mat / z
                    elif rule == 'ABS':
                        z = numpy.sum(contrib_mat, axis=0)
                        contrib = contrib_mat / abs(z)
                    elif rule == 'PN':
                        contrib_pos = numpy.where(contrib_mat >= 0, contrib_mat, 0)  # positive contributions
                        print (contrib_pos.shape)
                        contrib_pos = contrib_pos / contrib_pos.sum(0)  # positive contributions percentage
                        contrib_neg = numpy.where(contrib_mat < 0, contrib_mat, 0)  # negative contributions
                        contrib_neg = contrib_neg / (
                                    numpy.abs(contrib_neg.sum(0)) + 0.000001)  # negative contribution percentage
                        print(contrib_mat)
                        contrib_mat = contrib_neg + contrib_pos
                        print(contrib_mat)
                        return
                    # contrib_mat = contrib_mat / abs(contrib_mat).sum(axis=0)
                    contrib = contrib_mat.dot(c)
                    current_layer_contribs[j] = contrib
                    j += 1
                i-=1
                next_layer_contribs = current_layer_contribs
            return current_layer_contribs
        else :
            return self.compute_contrib_penultimate_layer(model,data)



    def compute_contrib_dense(self, model, layer_name, data, rule='L2'):
        n_chanels = len([layer for layer in model.layers if isinstance(layer, layers.InputLayer)])
        ow = model.get_layer("dense_2").get_weights()[0]
        dense1 = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        dense1_out = dense1.predict([data]*n_chanels)
        i = 0
        contribs = numpy.empty((dense1_out.shape[0],ow.shape[0], ow.shape[1]), dtype=numpy.float32)
        for out in dense1_out:
            out = out.reshape((out.shape[0],1))
            contrib = out * ow
            if rule == 'L2' : # Apply norm
                z = numpy.linalg.norm(contrib, ord=2, axis=0)
                contrib = contrib / z
            elif rule == 'L1' : # Apply L1-Norm
                z = numpy.linalg.norm(contrib, ord=1, axis=0)
                contrib = contrib / z
            elif rule == 'LRP-0' :
            # standard LRP -- uncomment the line below
                contrib = contrib / (numpy.sum(contrib,axis=0) + 0.0000000001)
            elif rule == 'PN' :
                contrib_pos = numpy.where(contrib>=0,contrib,0) # positive contributions
                contrib_pos = contrib_pos/(contrib_pos.sum(0)+0.00000001) # positive contributions percentage
                contrib_neg = numpy.where(contrib<0,contrib,0)
                contrib_neg = contrib_neg / (contrib_neg.sum(0)+0.0000001)
                contrib = contrib_neg + contrib_pos
            contribs[i] = contrib
            i += 1
        return contribs


    def compute_contrib_maxpool(self, model, layer_name, data, rule='L2'):
        n_chanels = len([layer for layer in model.layers if isinstance(layer, layers.InputLayer)])
        weights = model.get_layer('dense_1').get_weights()[0]
        c1 = self.compute_contrib_dense(model, "dense_1", data, rule)
        max_pool = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        max_out = max_pool.predict([data]*n_chanels)
        i = 0
        contribs = numpy.empty((max_out.shape[0], weights.shape[0], c1.shape[2]), dtype=numpy.float32)
        for (out, c) in zip(max_out, c1):
            out_1 = out.reshape((out.shape[0], 1))
            contrib_mat = out_1 * weights
            contrib = None
            if rule == 'L2' : # Apply norm
                z = numpy.linalg.norm(contrib_mat, ord=2, axis=0)
                contrib_mat = contrib_mat / z
            elif rule == 'L1' : # Apply L1-Norm
                z = numpy.linalg.norm(contrib_mat, ord=1, axis=0)
                contrib_mat = contrib_mat / z
            elif rule == 'LRP-0' :
            # standard LRP -- uncomment the line below
                z = numpy.sum(rule='L2',axis=0)
                contrib_mat = contrib_mat / z
            elif rule == 'PN' :
                contrib_pos = numpy.where(contrib_mat>=0,contrib_mat,0) # positive contributions
                contrib_pos = contrib_pos/(contrib_pos.sum(0)+0.00001) # positive contributions percentage
                contrib_neg = numpy.where(contrib_mat<0,contrib_mat,0) #negative contribution
                contrib_neg = contrib_neg / (numpy.abs(contrib_neg.sum(0))+0.000001) #negative contribution percentage
                contrib_mat = contrib_neg + contrib_pos

            #contrib_mat = contrib_mat / abs(contrib_mat).sum(axis=0)
            contrib = contrib_mat.dot(c)
            contribs[i] = contrib
            i += 1
        return contribs


    def compute_contributions(self,model, data, rule='L2'):
        c2 = self.compute_contrib_pooling_layer(model,data,rule)
        #c2 = self.compute_contrib_maxpool(model, self.max_pooled, data, rule)
        return c2

    """
           This method takes as input a sentence and a text cnn model and compute the necessary set of positive ngrams which 
           explain the model decision
    """
    def necessary_feature_set(self,model, sample, rule='L2'):
        sample = sample.reshape(1, len(sample))
        start = 0
        n_chanels = len([layer for layer in model.layers if isinstance(layer, layers.InputLayer)])
        contributions = self.compute_contributions(model, sample, rule)[0]
        ngrams = dict()
        conv_layers = [layer for layer in model.layers if isinstance(layer,layers.Conv1D)]
        for conv_layer in conv_layers:
            intermediate_layer_model = Model(inputs=model.input,
                                             outputs=conv_layer.output)
            intermediate_output = intermediate_layer_model.predict([sample]*n_chanels)
            n_filters = intermediate_output[0].shape[1]
            filter_size = conv_layer.kernel_size[0]
            out = intermediate_output[0]
            ngrams_indices = numpy.argmax(out, axis=0)  # indices of ngrams selected by global maxpooling.
            seq = [sample[0, t:t + filter_size] for t in ngrams_indices]
            filtered_ngrams = self.tokenizer.sequences_to_texts(seq)
            # compute the adjacency matrix : two filter are adjacents if they select the same ngram
            for i in range(n_filters):
                contrib = contributions[start + i]
                filters = [start + i]
                if filtered_ngrams[i] in ngrams:
                    filters += ngrams.get(filtered_ngrams[i]).get("filters")
                    contrib += ngrams.get(filtered_ngrams[i]).get("contrib")
                ngrams.update({filtered_ngrams[i]: {'filters': filters, 'contrib': contrib}})

            start += n_filters  # jump to the next list of filter (of different size)

        output_prob = model.predict([sample]*n_chanels)
        pred_class = numpy.argmax(output_prob)
        positive_ngrams = [(x[0], x[1], {
            'relevance': x[1]['contrib'][pred_class] - numpy.mean(numpy.delete(x[1]['contrib'], pred_class))}) for x in
                           ngrams.items() if x[1]['contrib'][pred_class] - numpy.mean(numpy.delete(x[1]['contrib'], pred_class))>0]
        positive_ngrams.sort(key=lambda tup: tup[2]['relevance'], reverse=True)
        new_model = model_from_json(model.to_json())
        new_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        new_model.set_weights(model.get_weights())
        i = 0
        retain_list = []
        dense_layers = [x for x in model.layers[::-1] if isinstance(x, layers.Dense)]
        first_dense_layer = dense_layers[-1]
        for ngram in positive_ngrams:
            maxpooled_value_weights = model.get_layer(first_dense_layer.name).get_weights()
            filters = ngram[1]['filters'] # all the filters associated to the courrent ngram
            for k in filters:
                maxpooled_value_weights[0][k] = 0;

            new_model.get_layer(first_dense_layer.name).set_weights(maxpooled_value_weights)
            y = new_model.predict([sample]*n_chanels)
            y = numpy.argmax(y)
            if pred_class != y:
                retain_list.append(ngram)

        necessary_features = {}
        for ngram in retain_list:
            token = ngram[0]
            key = str(len(token.split())) + '-ngrams'
            if key in necessary_features:
                necessary_features.get(key).append({ngram[0]: ngram[2]['relevance'].item()})
            else:
                necessary_features.update({key: [{ngram[0]: ngram[2]['relevance'].item()}]})

        return necessary_features

    """
        This method takes as input a sentence and a text cnn model and compute the sufficient set of positive ngrams which 
        explains the model decision
    """
    def sufficient_feature_set(self,model, sample, rule='L2'):
        sample = sample.reshape(1,len(sample))
        start = 0
        n_chanels = len([layer for layer in model.layers if isinstance(layer, layers.InputLayer)])
        contributions = self.compute_contributions(model, sample, rule)[0]
        ngrams = dict()
        conv_layers = [layer for layer in model.layers if isinstance(layer, layers.Conv1D)]
        for conv_layer in conv_layers :
            intermediate_layer_model = Model(inputs=model.input,
                                             outputs=conv_layer.output)
            intermediate_output = intermediate_layer_model.predict([sample]*n_chanels)
            #print(intermediate_output.shape)
            filter_size = conv_layer.kernel_size[0]
            n_filters = intermediate_output[0].shape[1]
            out = intermediate_output[0]
            ngrams_indices = numpy.argmax(out,axis = 0) #indices of ngrams selected by global maxpooling.
            seq = [sample[0,t:t + filter_size] for t in ngrams_indices]
            filtered_ngrams = self.tokenizer.sequences_to_texts(seq)
            #compute the adjacency matrix : two filter are adjacents if they select the same ngram
            for i in range(n_filters) :
                contrib = contributions[start+i]
                filters = [start+i]
                if filtered_ngrams[i] in ngrams :
                    filters += ngrams.get(filtered_ngrams[i]).get("filters")
                    contrib += ngrams.get(filtered_ngrams[i]).get("contrib")
                ngrams.update({filtered_ngrams[i]:{'filters':filters,'contrib':contrib}})

            start+=n_filters #jump to the next list of filter (of different size)

        output_prob = model.predict([sample]*n_chanels)
        pred_class = numpy.argmax(output_prob)
        positive_ngrams = [(x[0],x[1],{'relevance':x[1]['contrib'][pred_class]-numpy.mean(numpy.delete(x[1]['contrib'], pred_class))})
                           for x in ngrams.items() if x[1]['contrib'][pred_class]-numpy.mean(numpy.delete(x[1]['contrib'], pred_class))>0]
        positive_ngrams.sort(
            key=lambda tup: tup[2]['relevance'])
        # load weights into new model
        #new_model.load_weights(self.model_file_path + '.h5')
        new_model = model_from_json(model.to_json())
        new_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        new_model.set_weights(model.get_weights())
        i = 0
        drop_list = []
        #print(positive_ngrams)
        dense_layers = [x for x in model.layers[::-1] if isinstance(x, layers.Dense)]
        first_dense_layer = dense_layers[-1]
        for ngram in positive_ngrams : # activate progressively positive features and see which are sufficient
            filters = ngram[1]['filters']
            weights = new_model.get_layer(first_dense_layer.name).get_weights()
            for k in filters:
                    weights[0][k] = 0;
            new_model.get_layer(first_dense_layer.name).set_weights(weights)
            y = new_model.predict([sample]*n_chanels)
            y = numpy.argmax(y)
            if pred_class != y :
                break
            drop_list.append(ngram)
            i += 1

        sufficient_features = dict()
        for ngram in positive_ngrams :
            if ngram not in drop_list :
                token = ngram[0]
                key = str(len(token.split()))+'-ngrams'
                if key in sufficient_features :
                    sufficient_features.get(key).append({ngram[0]:ngram[2]['relevance'].item()})
                else :
                    sufficient_features.update({key:[{ngram[0]:ngram[2]['relevance'].item()}]})

        return sufficient_features

    def compute_ngrams_contributions(self, model, data, targets = None, rule='L2'):
        start = 0
        start_time = time.time()
        contribs = self.compute_contributions(model, data, rule)
        print("--- %s seconds ---" % (time.time() - start_time))
        n_chanels = len([layer for layer in model.layers if isinstance(layer, layers.InputLayer)])
        output_prob = model.predict([data]*n_chanels)
        pred_classes = numpy.argmax(output_prob, axis=1)
        if targets is not None :
            target_classes = numpy.argmax(targets, axis=1)
        else :
            target_classes = [None]*len(pred_classes)
        explanations = []
        for d,y,p in zip(data,target_classes,pred_classes) :
            target_class = y.item() if y is not None else None
            pred_class = p.item()
            if self.class_names is not None :
                if target_class is not None :
                    target_class = self.class_names[y]
                else :
                    target_class = None
                pred_class = self.class_names[p]

            e = {
                'sentence': self.tokenizer.sequences_to_texts([d]),
                'target_class': target_class,
                'predicted_class': pred_class,
                'features': {
                    'all': {},
                    #'sufficient':self.sufficient_feature_set(model,d),
                    #'necessary':self.necessary_feature_set(model,d)
                    'sufficient':[],
                    'necessary':[]
                }

            }
            explanations.append(e)
        start_time = time.time()
        conv_layers = [layer for layer in model.layers if isinstance(layer, layers.Conv1D)]
        for conv_layer in conv_layers :
            intermediate_layer_model = Model(inputs=model.input,
                                         outputs=conv_layer.output)
            filter_size = conv_layer.kernel_size[0]
            intermediate_output = intermediate_layer_model.predict([data]*n_chanels)
            n_filters = intermediate_output[0].shape[1]
            k = 0
            for (c_out, d, y, p) in zip(intermediate_output, data, target_classes, pred_classes):
                max_indices = numpy.argmax(c_out, axis=0)
                seq = [d[t:t+filter_size] for t in max_indices]
                filtered_ngrams = self.tokenizer.sequences_to_texts(seq)
                for i in range(n_filters):
                    contrib = contribs[k,start + i]
                    if filtered_ngrams[i] in explanations[k]['features']['all']:
                        contrib += explanations[k]['features']['all'].get(filtered_ngrams[i])
                    explanations[k]['features']['all'].update({filtered_ngrams[i]:contrib})

                k += 1
            start+=n_filters
        print("--- %s seconds ---" % (time.time() - start_time))
        for e, p in zip(explanations,pred_classes) :
            ngrams = dict()
            for key in e['features']['all'] :
                l_key = str(len(key.split())) + '-ngrams' #1-grams, 2-grams, 3-grams, etc.
                contrib  = e['features']['all'][key]
                #print("Contribution", key, contrib)
                rel = contrib[p]-numpy.mean(numpy.delete(contrib, p))
                contrib = [v.item() for v in contrib]
                if self.class_names is None :
                    contrib_dict = dict(zip(range(len(contrib)), contrib))
                else :
                    contrib_dict = dict(zip(self.class_names, contrib))
                contrib_dict.update({'Overall':rel.item()})
                if l_key in ngrams :
                    ngrams.get(l_key).append({key:contrib_dict})
                else :
                    ngrams.update({l_key:[{key: contrib_dict}]})
            e['features']['all'] = ngrams
        return explanations