import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';

class CNNModel {
    constructor(numClasses, imageSize = 128) {
        this.numClasses = numClasses;
        this.imageSize = imageSize;
        this.model = null;
        this.history = null;
    }

    createModel() {
        const model = tf.sequential({
            layers: [
                // First convolutional layer
                tf.layers.conv2d({
                    inputShape: [this.imageSize, this.imageSize, 3],
                    filters: 32,
                    kernelSize: 3,
                    activation: 'relu'
                }),
                tf.layers.maxPooling2d({ poolSize: 2 }),
                
                // Second convolutional layer
                tf.layers.conv2d({
                    filters: 64,
                    kernelSize: 3,
                    activation: 'relu'
                }),
                tf.layers.maxPooling2d({ poolSize: 2 }),
                
                // Third convolutional layer
                tf.layers.conv2d({
                    filters: 64,
                    kernelSize: 3,
                    activation: 'relu'
                }),
                tf.layers.maxPooling2d({ poolSize: 2 }),
                
                // Flatten and dense layers
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

    async train(X_train, y_train, X_test, y_test, epochs = 20, callbacks = {}) {
        if (!this.model) {
            throw new Error('Model not created. Call createModel() first.');
        }

        this.history = await this.model.fit(X_train, y_train, {
            epochs,
            validationData: [X_test, y_test],
            batchSize: 32,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    if (callbacks.onEpochEnd) {
                        callbacks.onEpochEnd(epoch, logs);
                    }
                },
                onTrainEnd: () => {
                    if (callbacks.onTrainEnd) {
                        callbacks.onTrainEnd();
                    }
                }
            }
        });

        return this.history;
    }

    async predict(X) {
        if (!this.model) {
            throw new Error('Model not trained');
        }
        return this.model.predict(X);
    }

    async evaluate(X_test, y_test) {
        if (!this.model) {
            throw new Error('Model not trained');
        }
        
        const evaluation = this.model.evaluate(X_test, y_test);
        const loss = await evaluation[0].data();
        const accuracy = await evaluation[1].data();
        
        evaluation[0].dispose();
        evaluation[1].dispose();
        
        return {
            loss: loss[0],
            accuracy: accuracy[0]
        };
    }

    async computeMetrics(X_test, y_test, reverseLabelMap) {
        const predictions = await this.predict(X_test);
        const predLabels = predictions.argMax(-1);
        const trueLabels = y_test.argMax(-1);
        
        const predArray = await predLabels.array();
        const trueArray = await trueLabels.array();
        
        // Compute confusion matrix and per-class accuracy
        const numClasses = this.numClasses;
        const confusionMatrix = Array(numClasses).fill().map(() => Array(numClasses).fill(0));
        const classCorrect = Array(numClasses).fill(0);
        const classTotal = Array(numClasses).fill(0);
        
        for (let i = 0; i < predArray.length; i++) {
            const trueLabel = trueArray[i];
            const predLabel = predArray[i];
            
            confusionMatrix[trueLabel][predLabel]++;
            classTotal[trueLabel]++;
            if (trueLabel === predLabel) {
                classCorrect[trueLabel]++;
            }
        }
        
        const perClassAccuracy = classCorrect.map((correct, idx) => 
            classTotal[idx] > 0 ? correct / classTotal[idx] : 0
        );
        
        // Rank diseases by accuracy
        const diseaseRanking = perClassAccuracy.map((acc, idx) => ({
            disease: reverseLabelMap.get(idx),
            accuracy: acc,
            samples: classTotal[idx]
        })).sort((a, b) => b.accuracy - a.accuracy);
        
        predictions.dispose();
        predLabels.dispose();
        trueLabels.dispose();
        
        return {
            confusionMatrix,
            perClassAccuracy,
            diseaseRanking,
            predictions: predArray,
            trueLabels: trueArray
        };
    }

    async saveModel() {
        if (!this.model) {
            throw new Error('No model to save');
        }
        
        const saveResult = await this.model.save('indexeddb://chicken-disease-model');
        return saveResult;
    }

    async loadModel() {
        try {
            this.model = await tf.loadLayersModel('indexeddb://chicken-disease-model');
            return true;
        } catch (error) {
            console.log('No saved model found');
            return false;
        }
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
        }
    }
}

export default CNNModel;