import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';

class DataLoader {
    constructor() {
        this.images = [];
        this.labels = [];
        this.labelMap = new Map();
        this.reverseLabelMap = new Map();
        this.imageSize = 128;
    }

    async loadDataset(zipFile, csvFile) {
        try {
            // Parse CSV file first to get label mapping
            await this.parseCSV(csvFile);
            
            // Extract and process images from ZIP
            await this.extractImages(zipFile);
            
            // Preprocess data
            return this.preprocessData();
        } catch (error) {
            console.error('Error loading dataset:', error);
            throw error;
        }
    }

    async parseCSV(csvFile) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const csvText = e.target.result;
                    const lines = csvText.split('\n').filter(line => line.trim());
                    
                    // Skip header if exists
                    const startIndex = lines[0].includes('image') || lines[0].includes('label') ? 1 : 0;
                    
                    // Create label mapping from all labels in CSV
                    const uniqueLabels = [...new Set(lines.slice(startIndex).map(line => {
                        const parts = line.split(',');
                        return parts[1] ? parts[1].trim() : '';
                    }))].filter(label => label);
                    
                    uniqueLabels.forEach((label, index) => {
                        this.labelMap.set(label, index);
                        this.reverseLabelMap.set(index, label);
                    });
                    
                    // Store filename to label mapping
                    this.fileLabelMap = new Map();
                    lines.slice(startIndex).forEach(line => {
                        const parts = line.split(',');
                        if (parts.length >= 2) {
                            const filename = parts[0].trim();
                            const label = parts[1].trim();
                            if (filename && label) {
                                this.fileLabelMap.set(filename, label);
                            }
                        }
                    });
                    
                    console.log(`Found ${this.fileLabelMap.size} labeled images`);
                    console.log(`Disease classes: ${Array.from(this.labelMap.keys())}`);
                    resolve();
                } catch (error) {
                    reject(error);
                }
            };
            reader.onerror = reject;
            reader.readAsText(csvFile);
        });
    }

    async extractImages(zipFile) {
        if (typeof JSZip === 'undefined') {
            throw new Error('JSZip library not loaded. Please include JSZip in your HTML.');
        }

        const zip = new JSZip();
        const zipContent = await zip.loadAsync(zipFile);
        
        this.images = [];
        this.labels = [];
        this.originalImages = [];
        
        // Process each file in ZIP
        const imageFiles = Object.keys(zipContent.files).filter(name => 
            name.match(/\.(jpg|jpeg|png)$/i) && !zipContent.files[name].dir
        );

        console.log(`Found ${imageFiles.length} images in ZIP`);

        let processedCount = 0;
        for (const filename of imageFiles) {
            const label = this.fileLabelMap.get(filename);
            if (label === undefined) {
                console.warn(`No label found for image: ${filename}`);
                continue;
            }

            try {
                const file = zipContent.files[filename];
                const blob = await file.async('blob');
                const img = await this.loadImage(blob);
                const processed = this.preprocessImage(img);
                
                this.images.push(processed);
                this.labels.push(this.labelMap.get(label));
                this.originalImages.push({
                    filename: filename,
                    label: label,
                    imageData: processed
                });
                
                processedCount++;
            } catch (error) {
                console.warn(`Failed to process image: ${filename}`, error);
            }
        }

        console.log(`Successfully processed ${processedCount} images`);
        
        if (processedCount === 0) {
            throw new Error('No images could be processed. Check if CSV filenames match ZIP filenames.');
        }
    }

    loadImage(blob) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = URL.createObjectURL(blob);
        });
    }

    preprocessImage(img) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = this.imageSize;
        canvas.height = this.imageSize;
        
        // Draw and resize image
        ctx.drawImage(img, 0, 0, this.imageSize, this.imageSize);
        return canvas;
    }

    preprocessData() {
        if (this.images.length === 0) {
            throw new Error('No images loaded');
        }

        // Convert to tensors
        const imageTensors = this.images.map(canvas => {
            return tf.browser.fromPixels(canvas)
                .toFloat()
                .div(255.0); // Normalize to [0, 1]
        });

        const X = tf.stack(imageTensors);
        const y = tf.oneHot(tf.tensor1d(this.labels, 'int32'), this.labelMap.size);

        // Clean up intermediate tensors
        imageTensors.forEach(t => t.dispose());

        // Split into train/test (80/20)
        const splitIndex = Math.floor(this.images.length * 0.8);
        const indices = tf.util.createShuffledIndices(this.images.length);
        
        const trainIndices = indices.slice(0, splitIndex);
        const testIndices = indices.slice(splitIndex);

        const X_train = tf.gather(X, trainIndices);
        const X_test = tf.gather(X, testIndices);
        const y_train = tf.gather(y, trainIndices);
        const y_test = tf.gather(y, testIndices);

        // Clean up
        X.dispose();
        y.dispose();

        return {
            X_train,
            X_test,
            y_train,
            y_test,
            labelMap: this.labelMap,
            reverseLabelMap: this.reverseLabelMap,
            originalImages: this.originalImages,
            trainIndices,
            testIndices
        };
    }

    dispose() {
        this.images = [];
        this.labels = [];
    }
}

export default DataLoader;
