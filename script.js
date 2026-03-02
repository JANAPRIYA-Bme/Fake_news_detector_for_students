// ---------- DARK MODE TOGGLE ----------

const toggle = document.getElementById("darkToggle");

if (toggle) {
    toggle.addEventListener("change", function () {
        document.body.classList.toggle("light-mode");
    });
}


// ---------- SMOOTH PROGRESS ANIMATION ----------

window.addEventListener("load", function () {
    const bar = document.querySelector(".progress-bar");

    if (bar) {
        const width = bar.style.width;
        bar.style.width = "0%";

        setTimeout(() => {
            bar.style.width = width;
        }, 300);
    }
});


// ---------- FEEDBACK BUTTONS ----------

const realBtn = document.querySelector(".real-btn");
const fakeBtn = document.querySelector(".fake-btn");

if (realBtn) {
    realBtn.addEventListener("click", () => {
        alert("Thank you for your feedback! Marked as Real.");
    });
}

if (fakeBtn) {
    fakeBtn.addEventListener("click", () => {
        alert("Thank you for your feedback! Marked as Fake.");
    });
}