import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';

class CNNModel {
    constructor(numClasses, imageSize = 128) {
        this.numClasses = numClasses;
        this.imageSize = imageSize;
        this.model = null;
    }

    createModel() {
        const model = tf.sequential({
            layers: [
                tf.layers.conv2d({
                    inputShape: [this.imageSize, this.imageSize, 3],
                    filters: 32,
                    kernelSize: 3,
                    activation: 'relu'
                }),
                tf.layers.maxPooling2d({ poolSize: 2 }),
                
                tf.layers.conv2d({
                    filters: 64,
                    kernelSize: 3,
                    activation: 'relu'
                }),
                tf.layers.maxPooling2d({ poolSize: 2 }),
                
                tf.layers.flatten(),
                tf.layers.dense({ units: 64, activation: 'relu' }),
                tf.layers.dropout({ rate: 0.5 }),
                tf.layers.dense({ units: this.numClasses, activation: 'softmax' })
            ]
        });

        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'categoricalCrossentropy',
            metrics: ['categoricalAccuracy']
        });

        this.model = model;
        return model;
    }

    async train(X_train, y_train, X_test, y_test, epochs = 15, callbacks = {}) {
        this.history = await this.model.fit(X_train, y_train, {
            epochs,
            validationData: [X_test, y_test],
            batchSize: 16,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    if (callbacks.onEpochEnd) {
                        callbacks.onEpochEnd(epoch, logs);
                    }
                }
            }
        });

        return this.history;
    }

    async predict(X) {
        return this.model.predict(X);
    }

    async computeMetrics(X_test, y_test, reverseLabelMap) {
        const predictions = await this.predict(X_test);
        const predLabels = predictions.argMax(-1);
        const trueLabels = y_test.argMax(-1);
        
        const predArray = await predLabels.array();
        const trueArray = await trueLabels.array();
        
        // Compute metrics
        const numClasses = this.numClasses;
        const classCorrect = Array(numClasses).fill(0);
        const classTotal = Array(numClasses).fill(0);
        
        for (let i = 0; i < predArray.length; i++) {
            const trueLabel = trueArray[i];
            const predLabel = predArray[i];
            
            classTotal[trueLabel]++;
            if (trueLabel === predLabel) {
                classCorrect[trueLabel]++;
            }
        }
        
        const perClassAccuracy = classCorrect.map((correct, idx) => 
            classTotal[idx] > 0 ? correct / classTotal[idx] : 0
        );
        
        const diseaseRanking = perClassAccuracy.map((acc, idx) => ({
            disease: reverseLabelMap.get(idx),
            accuracy: acc,
            samples: classTotal[idx]
        })).sort((a, b) => b.accuracy - a.accuracy);
        
        predictions.dispose();
        predLabels.dispose();
        trueLabels.dispose();
        
        return {
            perClassAccuracy,
            diseaseRanking,
            predictions: predArray,
            trueLabels: trueArray
        };
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
        }
    }
}

export default CNNModel;
