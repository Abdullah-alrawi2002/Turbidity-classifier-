import React, { useState } from "react";
import { StyleSheet, View, Text, Button, Image, ActivityIndicator, ScrollView, Alert } from "react-native";
import * as ImagePicker from "expo-image-picker";

const BACKEND_URL = "http://YOUR_BACKEND_IP:5000/predict"; // e.g., "http://192.168.1.100:5000/predict"

export default function App() {
  const [imageUri, setImageUri] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const pickImage = async () => {
    setResult(null);
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      Alert.alert("Permission required", "Media library access is needed.");
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      base64: false,
      quality: 0.8,
    });
    if (!result.cancelled) {
      setImageUri(result.uri);
      await uploadImage(result.uri);
    }
  };

  const takePhoto = async () => {
    setResult(null);
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      Alert.alert("Permission required", "Camera access is needed.");
      return;
    }
    const result = await ImagePicker.launchCameraAsync({
      base64: false,
      quality: 0.8,
    });
    if (!result.cancelled) {
      setImageUri(result.uri);
      await uploadImage(result.uri);
    }
  };

  const uploadImage = async (uri) => {
    setLoading(true);
    setResult(null);
    try {
      const form = new FormData();
      const filename = uri.split("/").pop();
      const match = /\.(\w+)$/.exec(filename);
      const type = match ? `image/${match[1]}` : "image/jpeg";
      form.append("image", {
        uri,
        name: filename,
        type,
      });
      const resp = await fetch(BACKEND_URL, {
        method: "POST",
        body: form,
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text);
      }
      const json = await resp.json();
      setResult(json);
    } catch (e) {
      Alert.alert("Upload failed", e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Water Turbidity Classifier</Text>
      <Text style={styles.subtitle}>Take or choose a water photo to get turbidity estimate.</Text>

      <View style={styles.buttonRow}>
        <Button title="Take Photo" onPress={takePhoto} />
        <View style={{ width: 12 }} />
        <Button title="Upload Image" onPress={pickImage} />
      </View>

      {imageUri && (
        <Image source={{ uri: imageUri }} style={styles.image} resizeMode="contain" />
      )}

      {loading && <ActivityIndicator size="large" style={{ marginTop: 20 }} />}

      {result && (
        <View style={styles.resultBox}>
          <Text style={styles.sectionTitle}>Prediction</Text>
          <Text>Class: <Text style={styles.bold}>{result.predicted_class}</Text></Text>
          <Text>Confidence: <Text style={styles.bold}>{(result.confidence * 100).toFixed(1)}%</Text></Text>
          <Text>
            NTU Range: <Text style={styles.bold}>{result.ntu_range[0]} – {result.ntu_range[1]}</Text>
          </Text>
          <Text style={{ marginTop: 8 }}>Per-class probabilities:</Text>
          {Object.entries(result.per_class_probs).map(([cls, p]) => (
            <Text key={cls}>
              {cls}: {(p * 100).toFixed(1)}%
            </Text>
          ))}
        </View>
      )}

      <Text style={styles.footer}>Ensure consistent lighting for best results.</Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 16,
    paddingBottom: 40,
    backgroundColor: "#f5f7fa",
  },
  title: {
    fontSize: 24,
    fontWeight: "600",
    marginBottom: 4,
    textAlign: "center",
  },
  subtitle: {
    fontSize: 14,
    color: "#555",
    marginBottom: 16,
    textAlign: "center",
  },
  buttonRow: {
    flexDirection: "row",
    justifyContent: "center",
    marginBottom: 12,
  },
  image: {
    width: "100%",
    height: 250,
    borderRadius: 8,
    marginTop: 12,
    backgroundColor: "#eee",
  },
  resultBox: {
    marginTop: 20,
    padding: 16,
    backgroundColor: "#ffffff",
    borderRadius: 8,
    shadowColor: "#000",
    shadowOpacity: 0.06,
    shadowRadius: 6,
    elevation: 3,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: "600",
    marginBottom: 8,
  },
  bold: {
    fontWeight: "700",
  },
  footer: {
    marginTop: 24,
    fontSize: 12,
    color: "#666",
    textAlign: "center",
  },
});
