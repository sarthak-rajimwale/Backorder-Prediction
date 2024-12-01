document.getElementById("predictionForm").addEventListener("submit", async function (e) {
    e.preventDefault();

    const formData = {
        features: [
            parseInt(document.getElementById("potential_issue").value),
            parseFloat(document.getElementById("perf_6_month_avg").value),
            parseFloat(document.getElementById("perf_12_month_avg").value),
            parseInt(document.getElementById("local_bo_qty").value),
            parseInt(document.getElementById("ppap_risk").value),
            parseFloat(document.getElementById("lead_time").value),
            parseInt(document.getElementById("deck_risk").value),
        ]
    };

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(formData),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const result = await response.json();

        if (result.prediction !== undefined) {
            document.getElementById("result").textContent = `Prediction: ${result.prediction}`;
        } else {
            document.getElementById("result").textContent = `Error: ${result.error || "Something went wrong"}`;
        }
    } catch (error) {
        document.getElementById("result").textContent = `Error: ${error.message}`;
    }
});
