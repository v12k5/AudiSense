🎙️ Sound Recorder App

A simple mobile app built with React Native (Expo) that lets you:

Record audio
Save the recording
Play the recording back

Works on both Android and iOS using the Expo Go app.

🚀 Features

Start / stop audio recording with one button
Save audio file internally (URI displayed on screen)
Play back the recording
Microphone permission handling
Cross-platform support with Expo

🛠️ Tech Stack

React Native

Expo

expo-av
 (record & play audio)

expo-file-system
 (file handling)

📂 Project Setup
1. Prerequisites

Node.js
 (LTS version recommended)

Expo Go app
 installed on your phone (Android/iOS)

Optional: Android Studio
 if you want to use an emulator

2. Install dependencies

Clone or download this project, then inside the folder run:

npm install

3. Run the app

Start the Expo dev server:

npx expo start


You’ll see a QR code in the terminal or Expo DevTools in the browser.

Scan the QR code with the Expo Go app on your phone.

The app will open instantly.

📱 Usage

Tap Start Recording → Speak something.
Tap Stop Recording → The file will be saved (you’ll see its URI).
Tap Play Recording → Hear your recorded audio.

📷 Screenshots (example layout)
+----------------------------------+
|   🎙️ Sound Recorder App          |
|                                  |
|   [ Start Recording ]            |
|                                  |
|   [ Play Recording ]             |
|   File saved at: file:///...     |
+----------------------------------+

⚠️ Notes

Audio files are stored in the app’s sandboxed storage (URI is shown).
On iOS, you must grant microphone permissions.
If using an Android emulator, make sure the microphone works in emulator settings.

📄 License

This project is open source and free to use.
