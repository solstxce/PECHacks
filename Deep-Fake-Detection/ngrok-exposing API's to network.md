# Ngrok Setup and Usage Guide for Exposing Local APIs

This guide will walk you through the steps to install **ngrok**, set it up, and connect it to your local FastAPI (or any web application) for public access.

## Prerequisites

- **ngrok** installed on your local machine
- **FastAPI** or any web server running on your local machine (we'll use FastAPI in this example)
- A running local application on port `8000` (or any other port of your choice)

## Step 1: Install Ngrok

### Windows
1. Go to the [ngrok website](https://ngrok.com/download) and download the Windows version of ngrok.
2. Extract the downloaded zip file.
3. Add the extracted `ngrok.exe` to your system's PATH (Optional: for easy access from anywhere in the terminal).

or 
Install ngrok via Chocolatey with the following command:
```
choco install ngrok
```
Run the following command to add your authtoken to the default ngrok.yml configuration file.
```
ngrok config add-authtoken 2qsNNZ96PS122QVGMA0b4floleh_59dHVUZhx5jRj7VYadQPX ( **You can directly do this instead of doing as a seperate command later again when setting it up in STEP-2**
```


### macOS/Linux
1. Download the ngrok binary using the following command in your terminal:

   ```bash
   brew install ngrok
   ```

   Alternatively, you can download it directly from the [ngrok website](https://ngrok.com/download).

2. Extract the file and move it to a directory of your choice (e.g., `/usr/local/bin`).

## Step 2: Set Up Ngrok

1. Sign up at [ngrok.com](https://ngrok.com/signup) and get your **auth token**.
2. Authenticate your ngrok client with the following command (replace `YOUR_AUTH_TOKEN` with your token):

   ```bash
   ngrok authtoken YOUR_AUTH_TOKEN
   ```

   This will link ngrok with your account and enable features like custom subdomains and additional connections.

## Step 3: Start Your Local API

Assuming you have a FastAPI app running on your local machine, start the app using `uvicorn`:

```bash
uvicorn your_app:app --reload --port 8000 (Specifying the port  using --port[] is optional)
```

By default, FastAPI runs on `http://localhost:8000`.(or at the very least in my case; You can specify the port that you want)

## Step 4: Expose Local API with Ngrok

1. Open a terminal and start ngrok by running the following command (make sure to use the same port that your FastAPI app is running on):

   ```bash
   ngrok http --url=stable-ideally-slug.ngrok-free.app 8000  
   ```

   - This command will create a secure tunnel to your localhost on port `8000` and provide you with a public URL.

2. After running the command, ngrok will display a URL similar to this:

   ```bash
   Forwarding                    https://1234abcd.ngrok-free.app -> http://localhost:8000
   ```

   This URL is now publicly accessible and points to your local API. You can use this URL to make API requests or share it with others.

## Step 5: Connect to Your API Publicly

1. **Test the Public URL**:
   - Open your browser or Postman, and enter the **public URL** (ex:`https://1234abcd.ngrok-free.app`) followed by the appropriate endpoint (e.g., `/detect` for your FastAPI app).

   Example:
   ```bash
   https://1234abcd.ngrok-free.app/detect
   ```

2. **Use the URL in Your Application**:
   - If your app is making requests to your local server, replace `http://localhost:8000` with the ngrok public URL `https://1234abcd.ngrok-free.app`.

## Step 6: Keeping Ngrok Running

- **Ngrok sessions will expire** after a certain period or when you close your terminal. If you need to keep it running, ensure the terminal stays open.
- You can also set up ngrok to run in the background on your server.

## Step 7: (Optional) Use a Custom Subdomain with Ngrok

If you have a **paid ngrok plan**, you can set a custom subdomain by running:

```bash
ngrok http -subdomain=myapp 8000
```

This will give you a URL like `https://myapp.ngrok.io`.

## Step 8: Security and Best Practices

- **Sensitive Data**: Exposing local servers to the internet via ngrok can be risky. Ensure that sensitive data or security-critical applications are protected.
- **Authentication**: Consider securing your FastAPI endpoints with authentication if you're exposing them to the public, especially when handling private or confidential data.

---

## Troubleshooting

1. **Port Already in Use**: If ngrok shows an error like `port already in use`, make sure your FastAPI app is running on the correct port, or choose a different port.
   
   ```bash
   ngrok http 8001
   ```

2. **Invalid URL**: If ngrok is showing an invalid URL after a few minutes, your free ngrok session might have expired. Run `ngrok` again to get a fresh URL.

---

## Conclusion

With **ngrok**, you can easily expose your local APIs or web apps to the public for testing or sharing. By following the steps above, you'll be able to quickly deploy your FastAPI (or other frameworks) and make it accessible on the internet with just a few commands.

