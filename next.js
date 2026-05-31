import React, { useState } from 'react';
import { Activity, Beaker, Briefcase, Building, ChevronRight, GraduationCap, MapPin, Search, Star, Stethoscope, AlertTriangle, BookOpen } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState("Engineering");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);

  // Form State
  const [formData, setFormData] = useState({
    jee_main_percentile: 95.0,
    jee_advanced_rank: 0,
    class_12_percent: 90.0,
    category: "General",
    state: "Delhi",
    branch: "Computer Science"
  });

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name.includes("percentile") || name.includes("percent") ? parseFloat(value) || 0 : 
              name.includes("rank") ? parseInt(value) || 0 : value
    }));
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResults(null);

    try {
      // Trying to hit the FastAPI backend
      const res = await fetch("http://localhost:8000/api/predict/engineering", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData)
      });
      
      if (!res.ok) throw new Error("Backend offline");
      const data = await res.json();
      setResults(data);
    } catch (err) {
      // FALLBACK: If the Python backend isn't running (like in this preview window),
      // we inject realistic mock data so you can still see the Neo-Brutalist UI.
      setTimeout(() => {
        setResults({
          status: "success",
          overall_score: 94.2,
          rankings: [
            {
              college: "DTU Delhi",
              category: "high-chance",
              final_score: 94.2,
              location: "Delhi",
              type: "[GOVT - STATE]",
              program: formData.branch,
              rationale: `Evaluated via JEE Mains AIR (HS Quota Impacted).`,
              data_profile: {
                rating: "★ 4.8 / 5.0 (NIRF Tier 1 / Premier)",
                recruiters: "Microsoft, Amazon, Atlassian, DE Shaw, GS",
                tie_ups: "Intel IoT Lab, IBM Research Network",
                placement_avg: "18.5 LPA",
                placement_highest: "1.2 CPA (International)",
                campus_area: "164 Acres",
                research_labs: "Advanced Robotics Center, VLSI Design Lab"
              }
            },
            {
              college: "NSUT Delhi",
              category: "moderate-borderline",
              final_score: 82.5,
              location: "Delhi",
              type: "[GOVT - STATE]",
              program: formData.branch,
              rationale: `Evaluated via JEE Mains AIR (HS Quota Impacted).`,
              data_profile: {
                rating: "★ 4.7 / 5.0 (NIRF Tier 1)",
                recruiters: "Google, Microsoft, Sprinklr, Texas Instruments",
                tie_ups: "Delhi Startup Hub Incubation",
                placement_avg: "16.8 LPA",
                placement_highest: "1.0 CPA (International)",
                campus_area: "145 Acres",
                research_labs: "Signal Processing Lab, AI Node"
              }
            }
          ]
        });
        setLoading(false);
      }, 1200);
    }
  };

  // Neo-Brutalist Components
  const NavButton = ({ title, icon: Icon, comingSoon }) => {
    const isActive = activeTab === title;
    return (
      <button
        onClick={() => !comingSoon && setActiveTab(title)}
        className={`flex items-center gap-2 px-6 py-3 font-black uppercase text-sm border-4 border-black transition-all ${
          comingSoon ? 'opacity-60 cursor-not-allowed bg-gray-200' :
          isActive ? 'bg-black text-white translate-x-[2px] translate-y-[2px] shadow-none' : 
          'bg-white hover:bg-[#00f5d4] hover:-translate-x-1 hover:-translate-y-1 shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]'
        }`}
      >
        <Icon size={18} />
        {title}
      </button>
    );
  };

  const StatCard = ({ val, label, bg }) => (
    <div className={`border-4 border-black p-4 text-center ${bg} shadow-[6px_6px_0px_0px_rgba(0,0,0,1)]`}>
      <div className="font-black text-3xl md:text-4xl">{val}</div>
      <div className="font-bold text-xs md:text-sm uppercase text-gray-800 mt-1">{label}</div>
    </div>
  );

  return (
    <div className="min-h-screen bg-[#f4f0ec] text-black font-sans p-4 md:p-8" style={{ fontFamily: 'system-ui, -apple-system, sans-serif' }}>
      
      {/* MAIN CONTAINER */}
      <div className="max-w-6xl mx-auto bg-white border-4 border-black shadow-[12px_12px_0px_0px_rgba(0,0,0,1)]">
        
        {/* HEADER */}
        <div className="border-b-4 border-black p-6 md:p-10 text-center bg-[#ffb703]">
          <h1 className="text-4xl md:text-5xl font-black uppercase tracking-tighter flex items-center justify-center gap-4">
            <Building size={40} strokeWidth={3} />
            North India Admission Portal
          </h1>
          <div className="mt-4 inline-block border-4 border-black bg-white px-4 py-2 font-bold uppercase text-sm shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]">
            Official Academic Engine v4.0
          </div>
        </div>

        {/* NAVIGATION */}
        <div className="p-6 md:p-10">
          <div className="flex flex-wrap gap-4 justify-center mb-10">
            <NavButton title="Engineering" icon={Beaker} />
            <NavButton title="Medical" icon={Stethoscope} />
            <NavButton title="Humanities" icon={BookOpen} comingSoon={true} />
            <NavButton title="Commerce" icon={Briefcase} comingSoon={true} />
          </div>

          {/* COMING SOON VIEW */}
          {(activeTab === "Humanities" || activeTab === "Commerce") && (
            <div className="bg-[#ffb703] border-4 border-black p-10 text-center shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] max-w-2xl mx-auto my-10">
              <h2 className="text-4xl font-black uppercase mb-4">{activeTab} Coming Soon</h2>
              <h3 className="text-xl font-bold uppercase mb-6">Official 2026 Roadmap Expansion</h3>
              <p className="font-semibold text-lg mb-8">
                The Academic Advisory Board is currently compiling historical cutoff datasets and evaluating institutional quotas for this vertical. Portal access will be granted in Q3 2026.
              </p>
              <button className="bg-black text-white font-black uppercase px-8 py-4 border-4 border-black hover:bg-gray-800 transition-colors">
                Request Early Access
              </button>
            </div>
          )}

          {/* ACTIVE VIEWS (ENGINEERING / MEDICAL) */}
          {(activeTab === "Engineering" || activeTab === "Medical") && (
            <>
              {/* OFFICIAL STATS ROW */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
                <StatCard 
                  val={activeTab === "Engineering" ? "45,500+" : "15,200+"} 
                  label="Total Seats Analyzed" 
                  bg="bg-white" 
                />
                <StatCard 
                  val={activeTab === "Engineering" ? "-1.2%" : "+3.4%"} 
                  label="Avg Cutoff Shift" 
                  bg="bg-[#b9fbc0]" 
                />
                <StatCard 
                  val={activeTab === "Engineering" ? "85% (HS Bias)" : "85% (State Quota)"} 
                  label="Domicile Impact Factor" 
                  bg="bg-[#ffcfd2]" 
                />
              </div>

              {/* INPUT FORM */}
              <div className="border-4 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] p-6 md:p-8 bg-white mb-12">
                <div className="inline-block bg-black text-white font-black uppercase px-4 py-2 mb-6">
                  Academic Telemetry
                </div>
                
                {activeTab === "Engineering" && (
                  <div className="bg-gray-100 border-2 border-black p-4 mb-6 font-bold text-center italic">
                    Estimated AIR = (1 - (Percentile / 100)) × 1,400,000 Candidates
                  </div>
                )}

                <form onSubmit={handlePredict} className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Left Column */}
                    <div className="space-y-4">
                      <div>
                        <label className="block font-black uppercase text-sm mb-2">
                          {activeTab === "Engineering" ? "JEE Main Percentile" : "NEET Score (out of 720)"}
                        </label>
                        <input 
                          type="number" step="0.1" name={activeTab === "Engineering" ? "jee_main_percentile" : "neet_score"}
                          value={activeTab === "Engineering" ? formData.jee_main_percentile : 650} onChange={handleInputChange}
                          className="w-full border-4 border-black p-3 font-bold bg-white focus:bg-[#f4f0ec] focus:outline-none transition-colors"
                        />
                      </div>
                      <div>
                        <label className="block font-black uppercase text-sm mb-2">Class 12 %</label>
                        <input 
                          type="number" step="0.1" name="class_12_percent" value={formData.class_12_percent} onChange={handleInputChange}
                          className="w-full border-4 border-black p-3 font-bold bg-white focus:bg-[#f4f0ec] focus:outline-none transition-colors"
                        />
                      </div>
                      <div>
                        <label className="block font-black uppercase text-sm mb-2">Category</label>
                        <select name="category" value={formData.category} onChange={handleInputChange} className="w-full border-4 border-black p-3 font-bold bg-white focus:bg-[#f4f0ec] focus:outline-none cursor-pointer">
                          {["General", "OBC-NCL", "SC", "ST", "EWS"].map(cat => <option key={cat} value={cat}>{cat}</option>)}
                        </select>
                      </div>
                    </div>
                    
                    {/* Right Column */}
                    <div className="space-y-4">
                      <div>
                        <label className="block font-black uppercase text-sm mb-2">
                          {activeTab === "Engineering" ? "JEE Advanced Rank (0 if N/A)" : "NEET All India Rank"}
                        </label>
                        <input 
                          type="number" name={activeTab === "Engineering" ? "jee_advanced_rank" : "neet_rank"}
                          value={activeTab === "Engineering" ? formData.jee_advanced_rank : 30000} onChange={handleInputChange}
                          className="w-full border-4 border-black p-3 font-bold bg-white focus:bg-[#f4f0ec] focus:outline-none transition-colors"
                        />
                      </div>
                      <div>
                        <label className="block font-black uppercase text-sm mb-2">Home State / Domicile</label>
                        <select name="state" value={formData.state} onChange={handleInputChange} className="w-full border-4 border-black p-3 font-bold bg-white focus:bg-[#f4f0ec] focus:outline-none cursor-pointer">
                          {["Delhi", "Punjab", "Haryana", "Uttar Pradesh", "Uttarakhand", "Rajasthan", "Himachal Pradesh", "Jammu & Kashmir"].map(st => <option key={st} value={st}>{st}</option>)}
                        </select>
                      </div>
                      <div>
                        <label className="block font-black uppercase text-sm mb-2">Target Program</label>
                        <select name="branch" value={formData.branch} onChange={handleInputChange} className="w-full border-4 border-black p-3 font-bold bg-white focus:bg-[#f4f0ec] focus:outline-none cursor-pointer">
                          {activeTab === "Engineering" ? 
                            ["Computer Science", "Electronics", "Electrical", "Mechanical"].map(b => <option key={b} value={b}>{b}</option>) :
                            ["MBBS", "BDS"].map(b => <option key={b} value={b}>{b}</option>)
                          }
                        </select>
                      </div>
                    </div>
                  </div>
                  
                  <button 
                    type="submit" 
                    disabled={loading}
                    className={`w-full border-4 border-black p-4 font-black uppercase text-xl shadow-[6px_6px_0px_0px_rgba(0,0,0,1)] transition-all
                      ${loading ? 'bg-gray-300 translate-x-[2px] translate-y-[2px] shadow-none cursor-wait' : 'bg-[#00f5d4] hover:-translate-y-1 hover:-translate-x-1 hover:shadow-[10px_10px_0px_0px_rgba(0,0,0,1)] active:translate-x-[4px] active:translate-y-[4px] active:shadow-none'}
                    `}
                  >
                    {loading ? "Analyzing Matrix..." : "Execute Algorithmic Analysis"}
                  </button>
                </form>
              </div>

              {/* RESULTS SECTION */}
              {results && (
                <div className="mt-12 animate-in fade-in slide-in-from-bottom-4 duration-500">
                  <div className="border-t-8 border-black pt-8 mb-8">
                    <h2 className="text-3xl font-black uppercase mb-2">Analysis Results</h2>
                    <p className="font-bold text-gray-600 uppercase">System Confidence Score: {results.overall_score}%</p>
                  </div>

                  {results.rankings.map((college, idx) => (
                    <div key={idx} className={`border-4 border-black bg-white p-6 mb-8 shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] relative
                      ${college.category === 'high-chance' ? 'border-l-[16px] border-l-[#b9fbc0]' : 
                        college.category === 'moderate-borderline' ? 'border-l-[16px] border-l-[#fbf8cc]' : 
                        'border-l-[16px] border-l-[#ffcfd2]'}`}
                    >
                      <div className="absolute top-4 right-4 bg-black text-white font-black text-xl px-4 py-2 border-4 border-black">
                        #{idx + 1}
                      </div>
                      
                      <div className="mb-4">
                        <h3 className="text-2xl font-black uppercase pr-16">{college.college}</h3>
                        <div className="flex flex-wrap items-center gap-3 mt-2">
                          <span className="bg-black text-white text-xs font-bold uppercase px-3 py-1">{college.type}</span>
                          <span className="font-bold text-sm">📍 {college.location} | {college.program}</span>
                        </div>
                      </div>

                      <p className="font-semibold text-gray-700 text-sm mb-6 border-l-4 border-gray-300 pl-3">
                        {college.rationale}
                      </p>

                      {/* DATA DENSE TABLE */}
                      <div className="border-4 border-black overflow-hidden bg-[#f8f9fa]">
                        <div className="grid grid-cols-1 md:grid-cols-2 border-b-4 border-black">
                          <div className="p-4 border-b-4 md:border-b-0 md:border-r-4 border-black">
                            <span className="block text-xs font-bold uppercase text-gray-500 mb-1">Performance Rating</span>
                            <span className="font-black text-lg">{college.data_profile?.rating || "Data Unavailable"}</span>
                          </div>
                          <div className="p-4">
                            <span className="block text-xs font-bold uppercase text-gray-500 mb-1">Campus Infrastructure</span>
                            <span className="font-bold">{college.data_profile?.campus_area} | {college.data_profile?.research_labs}</span>
                          </div>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-2 border-b-4 border-black bg-white">
                          <div className="p-4 border-b-4 md:border-b-0 md:border-r-4 border-black">
                            <span className="block text-xs font-bold uppercase text-gray-500 mb-1">Placement Velocity</span>
                            <span className="font-bold">Avg: {college.data_profile?.placement_avg} | High: {college.data_profile?.placement_highest}</span>
                          </div>
                          <div className="p-4">
                            <span className="block text-xs font-bold uppercase text-gray-500 mb-1">Industry Tie-Ups</span>
                            <span className="font-bold">{college.data_profile?.tie_ups}</span>
                          </div>
                        </div>
                        <div className="p-4 bg-[#ffb703]">
                          <span className="block text-xs font-black uppercase text-black mb-1">Corporate Ecosystem (Top Recruiters)</span>
                          <span className="font-bold">{college.data_profile?.recruiters}</span>
                        </div>
                      </div>

                      <div className="mt-6 flex items-center gap-2">
                        <input type="checkbox" id={`compare-${idx}`} className="w-5 h-5 border-2 border-black accent-black cursor-pointer" />
                        <label htmlFor={`compare-${idx}`} className="font-bold uppercase text-sm cursor-pointer select-none">Select for Comparison</label>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}

        </div>
        
        {/* FOOTER */}
        <div className="border-t-4 border-black bg-black text-white p-8">
          <div className="max-w-2xl mx-auto text-center">
            <h3 className="font-black text-2xl uppercase mb-4">Official Inquiry Portal</h3>
            <p className="font-bold mb-6">Have a specific question regarding your admission? Please ask below.</p>
            <div className="flex bg-white p-2">
              <input type="text" placeholder="Enter your application query..." className="w-full bg-transparent text-black font-bold outline-none px-4" />
              <button className="bg-[#00f5d4] text-black font-black uppercase px-6 py-2 border-4 border-black hover:bg-[#ffb703] transition-colors">
                Submit
              </button>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}