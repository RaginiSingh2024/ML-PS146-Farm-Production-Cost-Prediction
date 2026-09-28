/**
 * Farm Production Cost Prediction
 * JavaScript Frontend Controller
 * Case Study 146 - Machine Learning Semester V - ITM Skills University
 */

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const predictBtn = document.getElementById('predictBtn');
    const resetBtn = document.getElementById('resetBtn');
    const alertBox = document.getElementById('alertBox');
    const btnText = predictBtn.querySelector('.btn-text');
    const btnSpinner = predictBtn.querySelector('.btn-spinner');
    
    const resultPlaceholder = document.getElementById('resultPlaceholder');
    const resultContent = document.getElementById('resultContent');
    const resTotalCost = document.getElementById('resTotalCost');
    const resCropMeta = document.getElementById('resCropMeta');
    const resCostPerAcre = document.getElementById('resCostPerAcre');
    
    // Preset definitions for immediate local fill
    const LOCAL_PRESETS = {
        wheat: {
            crop_type: 'Wheat',
            farm_area: 5.0,
            seed_cost: 9000,
            fertilizer_usage: 600,
            labor_requirements: 60,
            irrigation_cost: 12500,
            pesticide_usage: 12.5,
            machinery_cost: 18500,
            transportation_cost: 2800
        },
        rice: {
            crop_type: 'Rice',
            farm_area: 6.0,
            seed_cost: 13200,
            fertilizer_usage: 900,
            labor_requirements: 132,
            irrigation_cost: 39000,
            pesticide_usage: 24,
            machinery_cost: 22500,
            transportation_cost: 3800
        },
        cotton: {
            crop_type: 'Cotton',
            farm_area: 7.5,
            seed_cost: 27000,
            fertilizer_usage: 975,
            labor_requirements: 210,
            irrigation_cost: 24000,
            pesticide_usage: 56,
            machinery_cost: 27000,
            transportation_cost: 3600
        },
        sugarcane: {
            crop_type: 'Sugarcane',
            farm_area: 10.0,
            seed_cost: 45000,
            fertilizer_usage: 2200,
            labor_requirements: 350,
            irrigation_cost: 75000,
            pesticide_usage: 45,
            machinery_cost: 38000,
            transportation_cost: 8500
        },
        maize: {
            crop_type: 'Maize',
            farm_area: 4.0,
            seed_cost: 8000,
            fertilizer_usage: 440,
            labor_requirements: 56,
            irrigation_cost: 8800,
            pesticide_usage: 12,
            machinery_cost: 15000,
            transportation_cost: 2400
        }
    };

    // Helper: Show alert message
    function showAlert(msg, isError = true) {
        alertBox.textContent = msg;
        alertBox.className = `alert-box ${isError ? 'alert-error' : 'alert-success'}`;
        alertBox.style.display = 'block';
        alertBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // Helper: Hide alert
    function hideAlert() {
        alertBox.style.display = 'none';
        alertBox.textContent = '';
    }

    // Helper: Toggle loading state
    function setLoading(isLoading) {
        if (isLoading) {
            predictBtn.disabled = true;
            btnText.textContent = 'Calculating Model Inference...';
            btnSpinner.style.display = 'inline-block';
        } else {
            predictBtn.disabled = false;
            btnText.textContent = 'Predict Production Cost';
            btnSpinner.style.display = 'none';
        }
    }

    // Preset Button Click Handlers
    document.querySelectorAll('.preset-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const cropKey = btn.getAttribute('data-crop');
            const preset = LOCAL_PRESETS[cropKey];
            if (preset) {
                hideAlert();
                document.getElementById('crop_type').value = preset.crop_type;
                document.getElementById('farm_area').value = preset.farm_area;
                document.getElementById('seed_cost').value = preset.seed_cost;
                document.getElementById('fertilizer_usage').value = preset.fertilizer_usage;
                document.getElementById('labor_requirements').value = preset.labor_requirements;
                document.getElementById('irrigation_cost').value = preset.irrigation_cost;
                document.getElementById('pesticide_usage').value = preset.pesticide_usage;
                document.getElementById('machinery_cost').value = preset.machinery_cost;
                document.getElementById('transportation_cost').value = preset.transportation_cost;
                
                showAlert(`Autofilled realistic farm parameters for ${preset.crop_type} (${preset.farm_area} Acres). Click 'Predict Production Cost' or modify inputs.`, false);
            }
        });
    });

    // Reset Button Handler
    resetBtn.addEventListener('click', () => {
        form.reset();
        hideAlert();
        resultPlaceholder.style.display = 'block';
        resultContent.style.display = 'none';
    });

    // Form Submit Handler
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        hideAlert();

        // Client-side validation
        const cropType = document.getElementById('crop_type').value;
        const farmArea = parseFloat(document.getElementById('farm_area').value);
        const seedCost = parseFloat(document.getElementById('seed_cost').value);
        const fertilizerUsage = parseFloat(document.getElementById('fertilizer_usage').value);
        const laborReq = parseFloat(document.getElementById('labor_requirements').value);
        const irrigationCost = parseFloat(document.getElementById('irrigation_cost').value);
        const pesticideUsage = parseFloat(document.getElementById('pesticide_usage').value);
        const machineryCost = parseFloat(document.getElementById('machinery_cost').value);
        const transportationCost = parseFloat(document.getElementById('transportation_cost').value);

        if (!cropType) {
            showAlert('Please select a valid Crop Type.');
            return;
        }

        if (isNaN(farmArea) || farmArea <= 0) {
            showAlert('Farm Area must be a positive numeric value (> 0 acres).');
            return;
        }

        const costInputs = [
            { name: 'Seed Cost', val: seedCost },
            { name: 'Fertilizer Usage', val: fertilizerUsage },
            { name: 'Labor Requirements', val: laborReq },
            { name: 'Irrigation Cost', val: irrigationCost },
            { name: 'Pesticide Usage', val: pesticideUsage },
            { name: 'Machinery Cost', val: machineryCost },
            { name: 'Transportation Cost', val: transportationCost }
        ];

        for (const item of costInputs) {
            if (isNaN(item.val) || item.val < 0) {
                showAlert(`${item.name} cannot be negative.`);
                return;
            }
        }

        // Prepare JSON payload
        const payload = {
            crop_type: cropType,
            farm_area: farmArea,
            seed_cost: seedCost,
            fertilizer_usage: fertilizerUsage,
            labor_requirements: laborReq,
            irrigation_cost: irrigationCost,
            pesticide_usage: pesticideUsage,
            machinery_cost: machineryCost,
            transportation_cost: transportationCost
        };

        setLoading(true);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                showAlert(data.error || 'Server error occurred while estimating production cost.');
                return;
            }

            // Display results smoothly
            resTotalCost.textContent = data.formatted_cost;
            resCropMeta.textContent = `${data.crop_type} • ${data.farm_area} Acres Farm`;
            resCostPerAcre.textContent = data.formatted_cost_per_acre;

            resultPlaceholder.style.display = 'none';
            resultContent.style.display = 'block';

            // Smooth scroll to result on mobile
            if (window.innerWidth < 960) {
                document.getElementById('resultCard').scrollIntoView({ behavior: 'smooth' });
            }

        } catch (err) {
            showAlert('Network or connection error. Please ensure the Flask server is running.');
            console.error('Prediction request failed:', err);
        } finally {
            setLoading(false);
        }
    });
});
