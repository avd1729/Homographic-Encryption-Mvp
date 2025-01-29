import React, { useState } from "react";
import ProgressBar from "./ProgressForm";
import Recommendations from "./Recommendations";

const UploadForm = () => {
  const [edgesFile, setEdgesFile] = useState(null);
  const [userFeaturesFile, setUserFeaturesFile] = useState(null);
  const [songFeaturesFile, setSongFeaturesFile] = useState(null);
  const [progress, setProgress] = useState(0);
  const [recommendations, setRecommendations] = useState(null);
  const [errors, setErrors] = useState({});

  const validateFile = (file) => {
    return file && file.name.endsWith(".csv");
  };

  const handleFileChange = (e, setFile, fieldName) => {
    const file = e.target.files[0];
    if (validateFile(file)) {
      setFile(file);
      setErrors((prevErrors) => ({ ...prevErrors, [fieldName]: null }));
    } else {
      setErrors((prevErrors) => ({
        ...prevErrors,
        [fieldName]: "Please upload a valid CSV file.",
      }));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!edgesFile || !userFeaturesFile || !songFeaturesFile) {
      setErrors({
        edges: !edgesFile ? "Please upload a valid edges CSV file." : null,
        userFeatures: !userFeaturesFile
          ? "Please upload a valid user features CSV file."
          : null,
        songFeatures: !songFeaturesFile
          ? "Please upload a valid song features CSV file."
          : null,
      });
      return;
    }

    const formData = new FormData();
    formData.append("edges", edgesFile);
    formData.append("user_features", userFeaturesFile);
    formData.append("song_features", songFeaturesFile);

    setProgress(20);

    try {
      const response = await fetch("http://localhost:5000/recommend", {
        method: "POST",
        body: formData,
      });

      setProgress(60);

      if (response.ok) {
        const data = await response.json();
        setRecommendations(data);
      } else {
        const errorText = await response.text();
        throw new Error(
          `Server responded with ${response.status}: ${errorText}`
        );
      }

      setProgress(100);
      setTimeout(() => setProgress(0), 500);
    } catch (error) {
      setRecommendations(`Error: ${error.message}`);
      setProgress(0);
    }
  };

  return (
    <div className="container">
      <form onSubmit={handleSubmit}>
        <label htmlFor="edges">Upload Edges CSV:</label>
        <input
          type="file"
          id="edges"
          accept=".csv"
          onChange={(e) => handleFileChange(e, setEdgesFile, "edges")}
          required
        />
        {errors.edges && (
          <div className="validation-message">{errors.edges}</div>
        )}

        <label htmlFor="user_features">Upload User Features CSV:</label>
        <input
          type="file"
          id="user_features"
          accept=".csv"
          onChange={(e) =>
            handleFileChange(e, setUserFeaturesFile, "userFeatures")
          }
          required
        />
        {errors.userFeatures && (
          <div className="validation-message">{errors.userFeatures}</div>
        )}

        <label htmlFor="song_features">Upload Song Features CSV:</label>
        <input
          type="file"
          id="song_features"
          accept=".csv"
          onChange={(e) =>
            handleFileChange(e, setSongFeaturesFile, "songFeatures")
          }
          required
        />
        {errors.songFeatures && (
          <div className="validation-message">{errors.songFeatures}</div>
        )}

        <button type="submit">Get Recommendations</button>
        {progress > 0 && <ProgressBar progress={progress} />}
      </form>

      <Recommendations data={recommendations} />
    </div>
  );
};

export default UploadForm;
