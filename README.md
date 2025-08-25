# Carni Detect

Carni Detect is a machine learning-based API for classifying dog breeds from uploaded images. It uses a TensorFlow model with a ResNet50 backbone to predict the breed of a dog and provides additional information about the breed.

## Requirements

- Docker Desktop must be installed on your system.

## Getting Started

Follow these steps to set up and run the Carni Detect API:

### 1. Clone the Repository

```bash
git clone git@github.com:niels-2005/carni_detect.git
```

### 2. Navigate to the Project Directory

```bash
cd carni_detect
```

### 3. Build the Docker Image

```bash
docker build -t carni_detect .
```

### 4. Run the Docker Container

```bash
docker run -p 8000:8000 carni_detect:latest
```

### 5. Access the API

The API will be available at:

```
http://127.0.0.1:8000/predict
```

## Using the API

The `/predict` endpoint expects a `POST` request with the following:

- **Body**: A `form-data` containing:
  - **Key**: `file`
  - **Value**: The image file.

### Example Request

You can use tools like Postman or `curl` to test the API. Here's an example using `curl`:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@path_to_your_image.jpg"
```

### Example Response

If the prediction is successful and meets the confidence threshold, the response will look like this:

```json
{
  "success": 1,
  "name": "Golden Retriever",
  "link": "https://www.fci.be/de/nomenclature/GOLDEN-RETRIEVER-111.html"
}
```

If the confidence is below the threshold, the response will indicate failure:

```json
{
  "success": 0
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.