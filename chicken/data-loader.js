import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';

class DataLoader {
    constructor() {
        this.labelMap = new Map();
        this.reverseLabelMap = new Map();
        this.imageSize = 128;
    }

    async loadFromCSV(csvFile) {
        const csvData = await this.readCSV(csvFile);
        const images = [];
        const labels = [];

        // Load images from URLs
        for (const row of csvData) {
            try {
                const img = await this.loadImageFromUrl(row.url);
                const tensor = this.preprocessImage(img);
                images.push(tensor);
                labels.push(row.labelIndex);
            } catch (error) {
                console.warn(`Failed to load image: ${row.url}`, error);
            }
        }

        if (images.length === 0) {
            throw new Error('No images could be loaded from the URLs');
        }

        return this.createTrainTestSplit(images, labels);
    }

    async readCSV(csvFile) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const csvText = e.target.result;
                    const lines = csvText.split('\n').filter(line => line.trim());
                    
                    // Skip header if exists
                    const startIndex = lines[0].includes('image_url') ? 1 : 0;
                    
                    // Get unique labels and create mapping
                    const uniqueLabels = [...new Set(lines.slice(startIndex).map(line => {
                        const parts = line.split(',');
                        return parts[1] ? parts[1].trim() : '';
                    }))].filter(label => label);
                    
                    uniqueLabels.forEach((label, index) => {
                        this.labelMap.set(label, index);
                        this.reverseLabelMap.set(index, label);
                    });

                    // Parse CSV data
                    const data = [];
                    for (let i = startIndex; i < lines.length; i++) {
                        const [url, label] = lines[i].split(',');
                        if (url && label && this.labelMap.has(label.trim())) {
                            data.push({
                                url: url.trim(),
                                label: label.trim(),
                                labelIndex: this.labelMap.get(label.trim())
                            });
                        }
                    }
                    
                    resolve(data);
                } catch (error) {
                    reject(error);
                }
            };
            reader.onerror = reject;
            reader.readAsText(csvFile);
        });
    }

    loadImageFromUrl(url) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => resolve(img);
            img.onerror = () => reject(new Error(`Failed to load image: ${url}`));
            img.src = url;
        });
    }

    preprocessImage(img) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = this.imageSize;
        canvas.height = this.imageSize;
        
        ctx.drawImage(img, 0, 0, this.imageSize, this.imageSize);
        return tf.browser.fromPixels(canvas)
            .toFloat()
            .div(255.0);
    }

    createTrainTestSplit(images, labels) {
        const X = tf.stack(images);
        const y = tf.oneHot(tf.tensor1d(labels, 'int32'), this.labelMap.size);

        // Clean up individual tensors
        images.forEach(t => t.dispose());

        // Split data (80% train, 20% test)
        const splitIndex = Math.floor(images.length * 0.8);
        const indices = tf.util.createShuffledIndices(images.length);
        
        const trainIndices = indices.slice(0, splitIndex);
        const testIndices = indices.slice(splitIndex);

        const X_train = tf.gather(X, trainIndices);
        const X_test = tf.gather(X, testIndices);
        const y_train = tf.gather(y, trainIndices);
        const y_test = tf.gather(y, testIndices);

        X.dispose();
        y.dispose();

        return {
            X_train,
            X_test,
            y_train,
            y_test,
            labelMap: this.labelMap,
            reverseLabelMap: this.reverseLabelMap
        };
    }
}

export default DataLoader;
