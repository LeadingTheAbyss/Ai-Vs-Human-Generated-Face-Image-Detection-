import { useRef, useState } from "react";

export default function App() {
  const videoRef = useRef(null);
  const soundCloseTimeoutRef = useRef(null);
  const lastVolumeRef = useRef(0.5);

  const [reason, setReason] = useState("");
  const [isMuted, setIsMuted] = useState(true);
  const [brightness, setBrightness] = useState(0);
  const [volume, setVolume] = useState(0);
  const [brightnessPinned, setBrightnessPinned] = useState(false);
  const [brightnessHover, setBrightnessHover] = useState(false);
  const [soundPinned, setSoundPinned] = useState(false);
  const [soundHover, setSoundHover] = useState(false);

  const fileInputRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState("");
  const [confidence, setConfidence] = useState("");
  const [loading, setLoading] = useState(false);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult("");
    setConfidence("");
    setReason("");
  };

  const handleSubmit = async () => {
    if (!selectedFile) {
      alert("Please select an image first");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      setReason("");
      setLoading(true);
      setResult("");
      setConfidence("");

      const response = await fetch("http://127.0.0.1:8080/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        alert(data.error || "Prediction failed");
        return;
      }

      setResult(data.result);
      setConfidence(`${data.confidence}%`);
      setReason(data.reason || "");
    } catch (error) {
      console.error(error);
      alert("Could not connect to Flask backend");
    } finally {
      setLoading(false);
    }
  };

  const toggleSound = async () => {
    const video = videoRef.current;
    if (!video) return;

    if (volume > 0 && !isMuted) {
      lastVolumeRef.current = volume;
      video.volume = 0;
      video.muted = true;
      setVolume(0);
      setIsMuted(true);
      return;
    }

    const restoredVolume =
      lastVolumeRef.current > 0 ? lastVolumeRef.current : 0.5;
    video.volume = restoredVolume;
    video.muted = false;
    setVolume(restoredVolume);
    setIsMuted(false);

    try {
      await video.play();
    } catch {
    }
  };

  const onBrightnessChange = async (e) => {
    const value = Number(e.target.value);
    setBrightness(value);

    const video = videoRef.current;
    if (!video) return;

    if (value > 0) {
      try {
        await video.play();
      } catch {}
    }
  };

  const onVolumeChange = async (e) => {
    const value = Number(e.target.value);
    setVolume(value);

    if (value > 0) {
      lastVolumeRef.current = value;
    }

    const video = videoRef.current;
    if (!video) return;

    video.volume = value;
    const mutedNow = value === 0;
    video.muted = mutedNow;
    setIsMuted(mutedNow);

    if (!mutedNow) {
      try {
        await video.play();
      } catch {
      }
    }
  };
  const openSoundPanel = () => {
    if (soundCloseTimeoutRef.current) {
      clearTimeout(soundCloseTimeoutRef.current);
    }
    setSoundHover(true);
  };

  const closeSoundPanel = () => {
    soundCloseTimeoutRef.current = setTimeout(() => {
      setSoundHover(false);
    }, 220);
  };

  const showBrightnessPanel = brightnessPinned || brightnessHover;
  const showSoundPanel = soundPinned || soundHover;

  const revealProgress = Math.min(brightness / 1.6, 1);
  const introOpacity = Math.max(0, 1 - revealProgress * 1.35);
  const typingOpacity = Math.max(0, 1 - revealProgress * 1.55);
  const splitOpacity = Math.max(0, 1 - revealProgress * 1.45);

  return (
    <main
      style={{
        minHeight: "100dvh",
        width: "100vw",
        minWidth: "100vw",
        position: "relative",
        display: "flex",
        flexDirection: "column",
        alignItems: "stretch",
        padding: 0,
        margin: 0,
        overflowX: "hidden",
        overflowY: "auto",
        background:
          "radial-gradient(circle at top, #10131b 0%, #07090f 42%, #020308 100%)",
      }}
    >
      <style>{`
        @keyframes uploadGlowShift {
          0% { background-position: 0 0; }
          50% { background-position: 400% 0; }
          100% { background-position: 0 0; }
        }
        @keyframes typingLoop {
          0% { width: 0ch; }
          18% { width: 0ch; }
          55% { width: 76ch; }
          72% { width: 59ch; }
          100% { width: 0ch; }
        }
        @keyframes caretBlink {
          50% { border-color: transparent; }
        }
      `}</style>
      <video
        ref={videoRef}
        autoPlay
        muted={isMuted}
        loop
        playsInline
        onLoadedData={() => {
          const video = videoRef.current;
          if (!video) return;
          video.volume = volume;
          video.muted = isMuted;
        }}
        style={{
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
          objectFit: "cover",
          objectPosition: "center top",
          transform: "translateY(-4%)",
          filter: `brightness(${0.35 + brightness * 0.9})`,
          opacity: revealProgress,
          zIndex: 0,
          transition: "opacity 220ms ease, filter 220ms ease",
        }}
      >
        <source src="/vid_max.mp4" type="video/mp4" />
      </video>
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: `rgba(0, 0, 0, ${0.45 - revealProgress * 0.2})`,
          zIndex: 0,
          transition: "background 220ms ease",
        }}
      />
      <h1
        style={{
          position: "absolute",
          top: "28px",
          left: "50%",
          transform: "translateX(-50%)",
          zIndex: 3,
          margin: 0,
          width: "calc(100vw - 64px)",
          textAlign: "center",
          fontSize: "clamp(1.45rem, 2.7vw, 2.5rem)",
          fontWeight: 700,
          letterSpacing: "0.02em",
          color: "#f8fafc",
          textShadow: "0 6px 24px rgba(0,0,0,0.45)",
          lineHeight: 1.15,
        }}
      >
        AI vs Human Generated Face Image Detection
      </h1>
      <div
        style={{ position: "absolute", top: "16px", left: "16px", zIndex: 3 }}
        onMouseEnter={() => setBrightnessHover(true)}
        onMouseLeave={() => setBrightnessHover(false)}
      >
        <button
          type="button"
          onClick={() => setBrightnessPinned((v) => !v)}
          style={{
            width: "48px",
            height: "48px",
            padding: 0,
            border: "none",
            background: "transparent",
            display: "grid",
            placeItems: "center",
            cursor: "pointer",
            boxShadow: "none",
          }}
          aria-label="Toggle brightness controls"
          title="Brightness"
        >
          <img
            src="/brightness.jpeg"
            alt="Brightness"
            style={{
              width: "35px",
              height: "35px",
              objectFit: "contain",
              display: "block",
              filter:
                "drop-shadow(0 0 2px rgba(255,255,255,0.95)) drop-shadow(0 0 10px rgba(255,255,255,0.55))",
            }}
          />
        </button>
        {showBrightnessPanel ? (
          <div
            style={{
              marginTop: "8px",
              minWidth: "190px",
              padding: "10px 12px",
              borderRadius: "14px",
              border: "1px solid rgba(255,255,255,0.32)",
              background:
                "linear-gradient(135deg, rgba(255,255,255,0.16), rgba(255,255,255,0.06))",
              color: "#ffffff",
              backdropFilter: "blur(8px)",
              boxShadow:
                "0 0 20px rgba(255,255,255,0.12), inset 0 0 12px rgba(255,255,255,0.06)",
            }}
          >
            <div
              style={{
                fontSize: "0.7rem",
                letterSpacing: "0.07em",
                textTransform: "uppercase",
                opacity: 0.95,
                marginBottom: "6px",
              }}
            >
              Brightness
            </div>
            <input
              type="range"
              min="0"
              max="1.6"
              step="0.01"
              value={brightness}
              onChange={onBrightnessChange}
              style={{
                width: "100%",
                accentColor: "#ffffff",
                cursor: "pointer",
              }}
            />
          </div>
        ) : null}
      </div>
      <div
        style={{
          position: "absolute",
          top: "16px",
          right: "16px",
          zIndex: 3,
          width: "48px",
          height: "48px",
        }}
        onMouseEnter={openSoundPanel}
        onMouseLeave={closeSoundPanel}
      >
        <button
          type="button"
          onClick={toggleSound}
          style={{
            width: "48px",
            height: "48px",
            padding: 0,
            border: "none",
            background: "transparent",
            display: "grid",
            placeItems: "center",
            cursor: "pointer",
            boxShadow: "none",
          }}
          aria-label="Toggle sound"
          title="Sound"
        >
          <img
            src={isMuted ? "/mute.jpeg" : "/sound.jpeg"}
            alt="Sound"
            style={{
              width: "30px",
              height: "30px",
              objectFit: "contain",
              display: "block",
              filter:
                "drop-shadow(0 0 2px rgba(255,255,255,0.95)) drop-shadow(0 0 10px rgba(255,255,255,0.55))",
            }}
          />
        </button>
        {showSoundPanel ? (
          <div
            onMouseEnter={openSoundPanel}
            onMouseLeave={closeSoundPanel}
            style={{
              position: "absolute",
              top: "100%",
              marginTop: "2px",
              right: 0,
              minWidth: "190px",
              padding: "10px 12px",
              borderRadius: "14px",
              border: "1px solid rgba(255,255,255,0.32)",
              background:
                "linear-gradient(135deg, rgba(255,255,255,0.16), rgba(255,255,255,0.06))",
              color: "#ffffff",
              backdropFilter: "blur(8px)",
              boxShadow:
                "0 0 20px rgba(255,255,255,0.12), inset 0 0 12px rgba(255,255,255,0.06)",
            }}
          >
            <div
              style={{
                fontSize: "0.7rem",
                letterSpacing: "0.07em",
                textTransform: "uppercase",
                opacity: 0.95,
                marginBottom: "6px",
              }}
            >
              Sound
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={volume}
              onChange={onVolumeChange}
              style={{
                width: "100%",
                accentColor: "#ffffff",
                cursor: "pointer",
              }}
            />
          </div>
        ) : null}
      </div>
      <div
        style={{
          position: "relative",
          zIndex: 1,
          width: "100%",
          maxWidth: "none",
          display: "grid",
          justifyItems: "center",
          gap: "22px",
          paddingTop: "110px",
          paddingBottom: "140px",
          paddingLeft: "24px",
          paddingRight: "24px",
          boxSizing: "border-box",
          pointerEvents: "none",
        }}
      >
        <div
          style={{
            opacity: splitOpacity,
            transform: `translateY(${revealProgress * -16}px) scale(${1 - revealProgress * 0.03})`,
            transition: "opacity 220ms ease, transform 220ms ease",
            width: "100%",
          }}
        >
          <div
            style={{
              width: "100%",
              display: "grid",
              gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
              gap: "20px",
            }}
          >
            <div
              style={{
                borderRadius: "22px",
                overflow: "hidden",
                border: "1px solid rgba(255,255,255,0.12)",
                background: "rgba(255,255,255,0.04)",
                boxShadow: "0 16px 50px rgba(0,0,0,0.34)",
                backdropFilter: "blur(4px)",
              }}
            >
              <div
                style={{
                  padding: "12px 16px",
                  fontSize: "0.85rem",
                  fontWeight: 600,
                  letterSpacing: "0.08em",
                  textTransform: "uppercase",
                  color: "rgba(255,255,255,0.78)",
                  borderBottom: "1px solid rgba(255,255,255,0.10)",
                }}
              >
                AI Generated Image
              </div>
              <img
                src="/ai.jpeg"
                alt="AI generated example"
                style={{
                  display: "block",
                  width: "100%",
                  height: "clamp(220px, 32vw, 390px)",
                  objectFit: "cover",
                }}
              />
            </div>
            <div
              style={{
                borderRadius: "22px",
                overflow: "hidden",
                border: "1px solid rgba(255,255,255,0.12)",
                background: "rgba(255,255,255,0.04)",
                boxShadow: "0 16px 50px rgba(0,0,0,0.34)",
                backdropFilter: "blur(4px)",
              }}
            >
              <div
                style={{
                  padding: "12px 16px",
                  fontSize: "0.85rem",
                  fontWeight: 600,
                  letterSpacing: "0.08em",
                  textTransform: "uppercase",
                  color: "rgba(255,255,255,0.78)",
                  borderBottom: "1px solid rgba(255,255,255,0.10)",
                }}
              >
                Real Image
              </div>
              <img
                src="/real.jpeg"
                alt="Real image example"
                style={{
                  display: "block",
                  width: "100%",
                  height: "clamp(220px, 32vw, 390px)",
                  objectFit: "cover",
                }}
              />
            </div>
          </div>
        </div>
        <div
          style={{
            opacity: typingOpacity,
            transition: "opacity 220ms ease",
            minHeight: "26px",
            display: "flex",
            justifyContent: "center",
          }}
        >
          <span
            style={{
              display: "inline-block",
              overflow: "hidden",
              whiteSpace: "nowrap",
              width: "0ch",
              fontFamily: "Consolas, monospace",
              fontSize: "clamp(0.92rem, 1.3vw, 1.05rem)",
              color: "rgba(194, 201, 214, 0.85)",
              borderRight: "2px solid rgba(194, 201, 214, 0.85)",
              animation:
                "typingLoop 11s steps(59, end) infinite, caretBlink 0.85s step-end infinite",
            }}
          >
            Website to detect whether an image of a Human Face is Real or AI
            generated
          </span>
        </div>
      </div>

      <div
        style={{
          position: "relative",
          zIndex: 3,
          width: "min(320px, 82vw)",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: "14px",
          margin: "36px auto 80px auto",
          alignSelf: "center",
        }}
      >
        <button
          type="button"
          onClick={handleUploadClick}
          style={{
            position: "relative",
            width: "100%",
            padding: "14px 16px",
            borderRadius: "10px",
            border: "1px solid rgba(255,255,255,0.24)",
            background: "rgba(0,0,0,0.38)",
            color: "#f8fafc",
            fontSize: "0.95rem",
            fontWeight: 600,
            letterSpacing: "0.02em",
            cursor: "pointer",
            overflow: "hidden",
            transition:
              "transform 260ms ease-in, box-shadow 260ms ease-in, border-color 260ms ease-in, color 260ms ease-in",
            boxShadow: "0 8px 28px rgba(0,0,0,0.38)",
          }}
          onMouseEnter={(e) => {
            const glow = e.currentTarget.querySelector(".rainbow-glow");
            if (glow) glow.style.opacity = "1";
            const fills = e.currentTarget.querySelectorAll(".water-fill");
            fills.forEach((fill) => {
              fill.style.transform = "scaleX(1)";
            });
            e.currentTarget.style.transform = "translateY(-1px) scale(1.01)";
            e.currentTarget.style.boxShadow = "0 14px 34px rgba(0,0,0,0.55)";
            e.currentTarget.style.borderColor = "rgba(255,255,255,0.46)";
            e.currentTarget.style.color = "#05070b";
          }}
          onMouseLeave={(e) => {
            const glow = e.currentTarget.querySelector(".rainbow-glow");
            if (glow) glow.style.opacity = "0";
            const fills = e.currentTarget.querySelectorAll(".water-fill");
            fills.forEach((fill) => {
              fill.style.transform = "scaleX(0)";
            });
            e.currentTarget.style.transform = "translateY(0) scale(1)";
            e.currentTarget.style.boxShadow = "0 8px 28px rgba(0,0,0,0.38)";
            e.currentTarget.style.borderColor = "rgba(255,255,255,0.24)";
            e.currentTarget.style.color = "#f8fafc";
          }}
        >
          <span
            className="rainbow-glow"
            aria-hidden="true"
            style={{
              position: "absolute",
              top: "-2px",
              left: "-2px",
              width: "calc(100% + 4px)",
              height: "calc(100% + 4px)",
              borderRadius: "12px",
              zIndex: 0,
              opacity: 0,
              filter: "blur(5px)",
              background:
                "linear-gradient(45deg, #ff0000, #ff7300, #fffb00, #48ff00, #00ffd5, #002bff, #7a00ff, #ff00c8, #ff0000)",
              backgroundSize: "400%",
              animation: "uploadGlowShift 20s linear infinite",
              transition: "opacity 0.3s ease-in-out",
              pointerEvents: "none",
            }}
          />
          <span
            className="water-fill"
            data-side="left"
            aria-hidden="true"
            style={{
              position: "absolute",
              left: 0,
              top: 0,
              bottom: 0,
              width: "50%",
              zIndex: 1,
              transform: "scaleX(0)",
              transformOrigin: "left center",
              transition: "transform 320ms ease-in",
              background:
                "linear-gradient(90deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.55) 100%)",
            }}
          />
          <span
            className="water-fill"
            data-side="right"
            aria-hidden="true"
            style={{
              position: "absolute",
              right: 0,
              top: 0,
              bottom: 0,
              width: "50%",
              zIndex: 1,
              transform: "scaleX(0)",
              transformOrigin: "right center",
              transition: "transform 320ms ease-in",
              background:
                "linear-gradient(270deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.55) 100%)",
            }}
          />
          <span style={{ position: "relative", zIndex: 2 }}>
            Upload an image
          </span>
        </button>
        {previewUrl && (
          <img
            src={previewUrl}
            alt="Preview"
            style={{
              width: "200px",
              maxHeight: "200px",
              objectFit: "cover",
              borderRadius: "14px",
              border: "1px solid rgba(255,255,255,0.18)",
              boxShadow: "0 10px 30px rgba(0,0,0,0.35)",
              display: "block",
            }}
          />
        )}
        {previewUrl && (
          <button
            type="button"
            onClick={handleSubmit}
            style={{
              position: "relative",
              width: "100%",
              padding: "14px 16px",
              borderRadius: "10px",
              border: "1px solid rgba(255,255,255,0.24)",
              background: "rgba(0,0,0,0.38)",
              color: "#f8fafc",
              fontSize: "0.95rem",
              fontWeight: 600,
              letterSpacing: "0.02em",
              cursor: "pointer",
              overflow: "hidden",
              transition:
                "transform 260ms ease-in, box-shadow 260ms ease-in, border-color 260ms ease-in, color 260ms ease-in",
              boxShadow: "0 8px 28px rgba(0,0,0,0.38)",
            }}
            onMouseEnter={(e) => {
              const glow = e.currentTarget.querySelector(".rainbow-glow");
              if (glow) glow.style.opacity = "1";
              const fills = e.currentTarget.querySelectorAll(".water-fill");
              fills.forEach((fill) => {
                fill.style.transform = "scaleX(1)";
              });
              e.currentTarget.style.transform = "translateY(-1px) scale(1.01)";
              e.currentTarget.style.boxShadow = "0 14px 34px rgba(0,0,0,0.55)";
              e.currentTarget.style.borderColor = "rgba(255,255,255,0.46)";
              e.currentTarget.style.color = "#05070b";
            }}
            onMouseLeave={(e) => {
              const glow = e.currentTarget.querySelector(".rainbow-glow");
              if (glow) glow.style.opacity = "0";
              const fills = e.currentTarget.querySelectorAll(".water-fill");
              fills.forEach((fill) => {
                fill.style.transform = "scaleX(0)";
              });
              e.currentTarget.style.transform = "translateY(0) scale(1)";
              e.currentTarget.style.boxShadow = "0 8px 28px rgba(0,0,0,0.38)";
              e.currentTarget.style.borderColor = "rgba(255,255,255,0.24)";
              e.currentTarget.style.color = "#f8fafc";
            }}
          >
            <span
              className="rainbow-glow"
              aria-hidden="true"
              style={{
                position: "absolute",
                top: "-2px",
                left: "-2px",
                width: "calc(100% + 4px)",
                height: "calc(100% + 4px)",
                borderRadius: "12px",
                zIndex: 0,
                opacity: 0,
                filter: "blur(5px)",
                background:
                  "linear-gradient(45deg, #ff0000, #ff7300, #fffb00, #48ff00, #00ffd5, #002bff, #7a00ff, #ff00c8, #ff0000)",
                backgroundSize: "400%",
                animation: "uploadGlowShift 20s linear infinite",
                transition: "opacity 0.3s ease-in-out",
                pointerEvents: "none",
              }}
            />
            <span
              className="water-fill"
              data-side="left"
              aria-hidden="true"
              style={{
                position: "absolute",
                left: 0,
                top: 0,
                bottom: 0,
                width: "50%",
                zIndex: 1,
                transform: "scaleX(0)",
                transformOrigin: "left center",
                transition: "transform 320ms ease-in",
                background:
                  "linear-gradient(90deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.55) 100%)",
              }}
            />
            <span
              className="water-fill"
              data-side="right"
              aria-hidden="true"
              style={{
                position: "absolute",
                right: 0,
                top: 0,
                bottom: 0,
                width: "50%",
                zIndex: 1,
                transform: "scaleX(0)",
                transformOrigin: "right center",
                transition: "transform 320ms ease-in",
                background:
                  "linear-gradient(270deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.55) 100%)",
              }}
            />
            <span style={{ position: "relative", zIndex: 2 }}>
              {loading ? "Checking..." : "Submit"}
            </span>
          </button>
        )}
        {result && (
  <div
    style={{
      textAlign: "center",
      color: "#f8fafc",
      background: "rgba(0,0,0,0.34)",
      border: "1px solid rgba(255,255,255,0.14)",
      borderRadius: "14px",
      padding: "14px 16px",
      width: "100%",
      boxShadow: "0 8px 28px rgba(0,0,0,0.28)",
    }}
  >
    <div style={{ fontSize: "1.05rem", fontWeight: 700 }}>
      Result: {result}
    </div>

    <div
      style={{
        marginTop: "6px",
        fontSize: "0.95rem",
        color: "rgba(255,255,255,0.82)",
      }}
    >
      Confidence: {confidence}
    </div>

        {reason && (
          <div
            style={{
              marginTop: "10px",
              paddingTop: "10px",
              borderTop: "1px solid rgba(255,255,255,0.10)",
              fontSize: "0.9rem",
              lineHeight: 1.55,
              color: "rgba(255,255,255,0.74)",
              textAlign: "left",
            }}
          >
            {reason}
          </div>
        )}
      </div>
    )}
        <input
          type="file"
          accept="image/*"
          ref={fileInputRef}
          onChange={handleFileChange}
          style={{ display: "none" }}
        />
      </div>
    </main>
  );
}
