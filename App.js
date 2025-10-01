import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, Button } from 'react-native';
import { Audio } from 'expo-av';
import * as FileSystem from 'expo-file-system';

export default function App() {
  const [recording, setRecording] = useState(null);
  const [sound, setSound] = useState(null);
  const [recordingUri, setRecordingUri] = useState(null);

  // Cleanup sound when unmounted
  useEffect(() => {
    return sound
      ? () => {
          console.log('Unloading Sound');
          sound.unloadAsync();
        }
      : undefined;
  }, [sound]);

  // Start recording
  async function startRecording() {
    try {
      console.log('Requesting permissions..');
      const permission = await Audio.requestPermissionsAsync();

      if (permission.status !== 'granted') {
        alert('Permission to access microphone is required!');
        return;
      }

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      console.log('Starting recording..');
      const { recording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );
      setRecording(recording);
      console.log('Recording started');
    } catch (err) {
      console.error('Failed to start recording', err);
    }
  }

  // Stop recording
  async function stopRecording() {
    console.log('Stopping recording..');
    await recording.stopAndUnloadAsync();
    const uri = recording.getURI();
    console.log('Recording stopped and stored at', uri);

    // ✅ Instead of moving, just save the same URI
    setRecording(null);
    setRecordingUri(uri);
  }

  // Play recorded audio
  async function playSound() {
    console.log('Loading Sound from', recordingUri);
    const { sound } = await Audio.Sound.createAsync({ uri: recordingUri });
    setSound(sound);

    console.log('Playing Sound...');
    await sound.playAsync();
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>🎙️ Sound Recorder App</Text>

      <Button
        title={recording ? 'Stop Recording' : 'Start Recording'}
        onPress={recording ? stopRecording : startRecording}
      />

      {recordingUri && (
        <>
          <Button title="Play Recording" onPress={playSound} />
          <Text style={styles.uri}>File saved at: {recordingUri}</Text>
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#222',
    padding: 20,
  },
  title: {
    fontSize: 22,
    color: '#fff',
    marginBottom: 20,
  },
  uri: {
    fontSize: 12,
    color: '#aaa',
    marginTop: 10,
    textAlign: 'center',
  },
});
