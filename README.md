# heart_disease_prediction
<!DOCTYPE html>
<html>
<head>
<style>
  body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f4f4f4;
  }

  #header {
    background-color: #333;
    color: #fff;
    text-align: center;
    padding: 10px;
  }

  #container {
    width: 80%;
    margin: auto;
    padding: 20px;
    background-color: #fff;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
  }

  #models {
    margin-top: 20px;
  }

  table {
    border-collapse: collapse;
    width: 100%;
  }

  th, td {
    padding: 8px;
    text-align: left;
    border-bottom: 1px solid #ddd;
  }

  th {
    background-color: #f2f2f2;
  }

  .accuracy {
    font-weight: bold;
    color: green;
  }
</style>
</head>
<body>
  <div id="header">
    <h1>Heart Disease Prediction Project</h1>
  </div>
  <div id="container">
    <h2>Libraries Used</h2>
    <p>Python libraries used in this project:</p>
    <ul>
      <li>numpy</li>
      <li>pandas</li>
      <li>matplotlib.pyplot</li>
      <li>seaborn</li>
      <li>sklearn.linear_model.LogisticRegression</li>
      <!-- Add other library items here -->
    </ul>

    <h2>Dataset Columns</h2>
    <p>Columns in the dataset:</p>
    <table>
      <tr>
        <th>Column</th>
        <th>Non-Null Count</th>
        <th>Dtype</th>
      </tr>
      <!-- Add rows for each column here -->
    </table>

    <h2>Models Used</h2>
    <div id="models">
      <p>Classification models and their accuracy:</p>
      <ul>
        <li><strong>Logistic Regression:</strong> <span class="accuracy">88.52%</span></li>
        <li>K-Nearest Neighbors (KNN): <span class="accuracy">68.85%</span></li>
        <!-- Add other models and their accuracy here -->
      </ul>
    </div>

    <h2>Conclusion</h2>
    <p>Summary of the project and its findings.</p>
  </div>
</body>
</html>
