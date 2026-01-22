import React, { useState } from 'react';
import { Upload, Play, Download, Settings, Info } from 'lucide-react';

const SquirrelTracker = () => {
  const [selectedMethod, setSelectedMethod] = useState('yolo');
  const [videoFile, setVideoFile] = useState(null);
  const [videoLocation, setVideoLocation] = useState('inside');
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const methods = [
    { 
      id: 'yolo', 
      name: 'YOLO Tracking',
      description: 'Objekt-Erkennung und Tracking mit YOLO',
      suitable: ['inside', 'outside']
    },
    { 
      id: 'background_sub', 
      name: 'Background Subtraction',
      description: 'Bewegungserkennung durch Hintergrund-Subtraktion',
      suitable: ['inside', 'outside']
    },
    { 
      id: 'sam', 
      name: 'SAM (Segment Anything)',
      description: 'Segmentierung mit SAM Modell',
      suitable: ['outside']
    },
    { 
      id: 'optical_flow', 
      name: 'Optical Flow',
      description: 'Bewegungsanalyse mit optischem Fluss',
      suitable: ['inside', 'outside']
    },
    { 
      id: 'time_analysis', 
      name: 'Zeitanalyse',
      description: 'Aufenthaltsdauer in Box und Eingang',
      suitable: ['inside']
    }
  ];

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('video/')) {
      setVideoFile(file);
      setResults(null);
    }
  };

  const handleProcess = async () => {
    if (!videoFile) return;
    
    setIsProcessing(true);
    setProgress(0);
    setResults(null);
    
    try {
      const formData = new FormData();
      formData.append('video', videoFile);
      formData.append('method', selectedMethod);
      
      const response = await fetch('http://localhost:5000/analyze', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Analyse fehlgeschlagen');
      }
      
      const data = await response.json();
      
      setResults({
        method: selectedMethod,
        videoName: videoFile.name,
        duration: data.duration,
        framesProcessed: data.frames_processed,
        avgMovement: data.avg_movement?.toFixed(0),
        maxMovement: data.max_movement,
        peakFrame: data.peak_frame,
        detections: data.total_detections,
        plot: data.plot
      });
      
      setProgress(100);
    } catch (error) {
      console.error('Fehler:', error);
      alert('Fehler bei der Analyse: ' + error.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const getMethodInfo = (methodId) => {
    return methods.find(m => m.id === methodId);
  };

  const isMethodSuitable = (methodId) => {
    const method = methods.find(m => m.id === methodId);
    return method?.suitable.includes(videoLocation);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 to-teal-100 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-12 h-12 bg-emerald-600 rounded-lg flex items-center justify-center">
              <span className="text-2xl">🐿️</span>
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-800">Eichhörnchen-Tracking System</h1>
              <p className="text-gray-600">Videoanalyse für Verhaltensforschung</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Konfiguration */}
          <div className="lg:col-span-2 space-y-6">
            {/* Video Upload */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                <Upload className="w-5 h-5" />
                Video hochladen
              </h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Aufnahme-Ort
                  </label>
                  <div className="flex gap-3">
                    <button
                      onClick={() => setVideoLocation('inside')}
                      className={`flex-1 py-3 px-4 rounded-lg font-medium transition-all ${
                        videoLocation === 'inside'
                          ? 'bg-emerald-600 text-white shadow-md'
                          : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                      }`}
                    >
                      Innerhalb der Box
                    </button>
                    <button
                      onClick={() => setVideoLocation('outside')}
                      className={`flex-1 py-3 px-4 rounded-lg font-medium transition-all ${
                        videoLocation === 'outside'
                          ? 'bg-emerald-600 text-white shadow-md'
                          : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                      }`}
                    >
                      Außerhalb der Box
                    </button>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Video-Datei
                  </label>
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-emerald-500 transition-colors cursor-pointer">
                    <input
                      type="file"
                      accept="video/*"
                      onChange={handleFileUpload}
                      className="hidden"
                      id="video-upload"
                    />
                    <label htmlFor="video-upload" className="cursor-pointer">
                      <Upload className="w-12 h-12 mx-auto text-gray-400 mb-3" />
                      {videoFile ? (
                        <div>
                          <p className="text-emerald-600 font-medium">{videoFile.name}</p>
                          <p className="text-sm text-gray-500 mt-1">Klicken zum Ändern</p>
                        </div>
                      ) : (
                        <div>
                          <p className="text-gray-600 font-medium">Video hier ablegen oder klicken</p>
                          <p className="text-sm text-gray-500 mt-1">MP4, AVI, MOV, etc.</p>
                        </div>
                      )}
                    </label>
                  </div>
                </div>
              </div>
            </div>

            {/* Methoden-Auswahl */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                <Settings className="w-5 h-5" />
                Analysemethode
              </h2>
              
              <div className="space-y-3">
                {methods.map(method => {
                  const suitable = isMethodSuitable(method.id);
                  return (
                    <button
                      key={method.id}
                      onClick={() => suitable && setSelectedMethod(method.id)}
                      disabled={!suitable}
                      className={`w-full text-left p-4 rounded-lg border-2 transition-all ${
                        selectedMethod === method.id
                          ? 'border-emerald-600 bg-emerald-50'
                          : suitable
                          ? 'border-gray-200 hover:border-emerald-300'
                          : 'border-gray-100 bg-gray-50 opacity-50 cursor-not-allowed'
                      }`}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <h3 className="font-semibold text-gray-800">{method.name}</h3>
                          <p className="text-sm text-gray-600 mt-1">{method.description}</p>
                          {!suitable && (
                            <p className="text-xs text-amber-600 mt-2">
                              ⚠️ Nicht geeignet für {videoLocation === 'inside' ? 'Innen-' : 'Außen-'}Aufnahmen
                            </p>
                          )}
                        </div>
                        {selectedMethod === method.id && (
                          <div className="w-5 h-5 bg-emerald-600 rounded-full flex items-center justify-center ml-3">
                            <div className="w-2 h-2 bg-white rounded-full"></div>
                          </div>
                        )}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Verarbeitung & Ergebnisse */}
          <div className="space-y-6">
            {/* Verarbeitung starten */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <button
                onClick={handleProcess}
                disabled={!videoFile || isProcessing}
                className={`w-full py-4 rounded-lg font-semibold flex items-center justify-center gap-2 transition-all ${
                  !videoFile || isProcessing
                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                    : 'bg-emerald-600 text-white hover:bg-emerald-700 shadow-lg hover:shadow-xl'
                }`}
              >
                <Play className="w-5 h-5" />
                {isProcessing ? 'Wird verarbeitet...' : 'Analyse starten'}
              </button>

              {isProcessing && (
                <div className="mt-4">
                  <div className="flex justify-between text-sm text-gray-600 mb-2">
                    <span>Video wird analysiert...</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                    <div className="bg-emerald-600 h-full animate-pulse rounded-full w-full"></div>
                  </div>
                </div>
              )}
            </div>

            {/* Ergebnisse */}
            {results && (
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                  <Info className="w-5 h-5" />
                  Ergebnisse
                </h2>
                
                <div className="space-y-3">
                  <div className="bg-emerald-50 rounded-lg p-4">
                    <p className="text-sm text-gray-600">Methode</p>
                    <p className="font-semibold text-gray-800">{getMethodInfo(results.method)?.name}</p>
                  </div>
                  
                  <div className="bg-blue-50 rounded-lg p-4">
                    <p className="text-sm text-gray-600">Video</p>
                    <p className="font-semibold text-gray-800 text-sm">{results.videoName}</p>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-purple-50 rounded-lg p-4">
                      <p className="text-sm text-gray-600">Dauer</p>
                      <p className="font-semibold text-gray-800">{results.duration}</p>
                    </div>
                    
                    <div className="bg-orange-50 rounded-lg p-4">
                      <p className="text-sm text-gray-600">Frames</p>
                      <p className="font-semibold text-gray-800">{results.framesProcessed}</p>
                    </div>
                  </div>

                  {results.avgMovement && (
                    <div className="bg-teal-50 rounded-lg p-4">
                      <p className="text-sm text-gray-600">Ø Bewegung (Pixel)</p>
                      <p className="font-semibold text-gray-800">{results.avgMovement}</p>
                    </div>
                  )}

                  {results.detections !== undefined && (
                    <div className="bg-pink-50 rounded-lg p-4">
                      <p className="text-sm text-gray-600">Bewegungs-Events</p>
                      <p className="font-semibold text-gray-800">{results.detections}</p>
                    </div>
                  )}

                  {results.plot && (
                    <div className="mt-4">
                      <p className="text-sm font-medium text-gray-700 mb-2">Bewegungs-Analyse:</p>
                      <img 
                        src={`data:image/png;base64,${results.plot}`} 
                        alt="Analysis Plot"
                        className="w-full rounded-lg border border-gray-200"
                      />
                    </div>
                  )}
                </div>

                <button className="w-full mt-4 py-3 bg-gray-800 text-white rounded-lg font-medium flex items-center justify-center gap-2 hover:bg-gray-900 transition-colors">
                  <Download className="w-4 h-4" />
                  Ergebnisse exportieren
                </button>
              </div>
            )}

            {/* Info Box */}
            <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
              <h3 className="font-semibold text-blue-900 mb-2">💡 Hinweis</h3>
              <p className="text-sm text-blue-800">
                Wähle die passende Methode basierend auf deinem Aufnahme-Ort. 
                YOLO funktioniert am besten mit Trainingsdaten, SAM ist ideal für Außenaufnahmen.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SquirrelTracker;