from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# === Muat model regresi linear yang sudah dilatih ===
MODEL_PATH = "model_Calories_Burned.pkl"
model = joblib.load(MODEL_PATH)

# === Nilai evaluasi model dari Jupyter Notebook ===
R2_SCORE = 0.8312294819661906
RMSE = 111.0842560127687


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        prediction=None,
        r2=R2_SCORE,
        rmse=RMSE,
        duration_input=""
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil input durasi dari form
        duration_input = request.form.get("duration", "").strip()

        # Validasi input kosong
        if not duration_input:
            error_message = "Durasi tidak boleh kosong. Silakan isi durasi latihan dalam menit."
            return render_template(
                "index.html",
                error=error_message,
                r2=R2_SCORE,
                rmse=RMSE,
                prediction=None,
                duration_input=""
            )

        # Validasi input harus angka
        try:
            duration = float(duration_input)
        except ValueError:
            error_message = "Durasi harus berupa angka. Contoh: 30"
            return render_template(
                "index.html",
                error=error_message,
                r2=R2_SCORE,
                rmse=RMSE,
                prediction=None,
                duration_input=""
            )

        # Konversi menit ke jam (model dilatih dengan durasi dalam jam)
        duration_hours = duration / 60.0

        # Prediksi kalori
        prediction = model.predict(np.array([[duration_hours]]))[0]

        # Format hasil
        prediction_text = f"{prediction:.2f} kalori"

        return render_template(
            "index.html",
            prediction=prediction_text,
            r2=R2_SCORE,
            rmse=RMSE,
            duration_input=duration
        )

    except Exception:
        # Tangani error umum
        error_message = "Terjadi kesalahan: pastikan Anda mengisi durasi latihan dengan angka yang valid."
        return render_template(
            "index.html",
            error=error_message,
            r2=R2_SCORE,
            rmse=RMSE,
            prediction=None,
            duration_input=""
        )


if __name__ == "__main__":
    app.run(debug=True)
