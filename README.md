<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AGI Research Repository Cover</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
            overflow: hidden;
        }
        
        .cover-container {
            width: 1200px;
            height: 400px;
            position: relative;
            overflow: hidden;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        }
        
        .neural-network {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            opacity: 0.3;
        }
        
        .node {
            position: absolute;
            width: 8px;
            height: 8px;
            background: radial-gradient(circle, #64ffda 0%, #00bcd4 100%);
            border-radius: 50%;
            animation: pulse 2s infinite ease-in-out;
        }
        
        .node.large {
            width: 12px;
            height: 12px;
            background: radial-gradient(circle, #ff6b6b 0%, #ee5a24 100%);
        }
        
        .node.medium {
            width: 10px;
            height: 10px;
            background: radial-gradient(circle, #a8e6cf 0%, #4ecdc4 100%);
        }
        
        .connection {
            position: absolute;
            height: 1px;
            background: linear-gradient(90deg, transparent 0%, #64ffda 50%, transparent 100%);
            opacity: 0.4;
            animation: flow 3s infinite ease-in-out;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 0.7; }
            50% { transform: scale(1.2); opacity: 1; }
        }
        
        @keyframes flow {
            0%, 100% { opacity: 0.2; }
            50% { opacity: 0.6; }
        }
        
        .brain-outline {
            position: absolute;
            right: 50px;
            top: 50px;
            width: 200px;
            height: 150px;
            opacity: 0.2;
            filter: drop-shadow(0 0 20px #64ffda);
        }
        
        .content {
            position: relative;
            z-index: 10;
            padding: 60px;
            color: white;
        }
        
        .title {
            font-size: 42px;
            font-weight: 700;
            background: linear-gradient(135deg, #64ffda 0%, #ff6b6b 50%, #a8e6cf 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 15px;
            text-shadow: 0 0 30px rgba(100, 255, 218, 0.3);
        }
        
        .subtitle {
            font-size: 18px;
            color: #b0c4de;
            margin-bottom: 25px;
            font-weight: 300;
            letter-spacing: 1px;
        }
        
        .description {
            font-size: 14px;
            color: #8892b0;
            line-height: 1.6;
            max-width: 600px;
            margin-bottom: 30px;
        }
        
        .tech-stack {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        
        .tech-tag {
            background: rgba(100, 255, 218, 0.1);
            border: 1px solid rgba(100, 255, 218, 0.3);
            color: #64ffda;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }
        
        .tech-tag:hover {
            background: rgba(100, 255, 218, 0.2);
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(100, 255, 218, 0.2);
        }
        
        .floating-elements {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }
        
        .floating-icon {
            position: absolute;
            opacity: 0.1;
            animation: float 6s infinite ease-in-out;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(180deg); }
        }
        
        .github-corner {
            position: absolute;
            top: 20px;
            right: 20px;
            color: #64ffda;
            font-size: 24px;
            opacity: 0.7;
        }
        
        .research-badge {
            position: absolute;
            bottom: 20px;
            right: 20px;
            background: rgba(255, 107, 107, 0.1);
            border: 1px solid rgba(255, 107, 107, 0.3);
            color: #ff6b6b;
            padding: 10px 20px;
            border-radius: 25px;
            font-size: 12px;
            font-weight: 600;
            backdrop-filter: blur(10px);
        }
        
        .grid-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            opacity: 0.05;
            background-image: 
                linear-gradient(rgba(100, 255, 218, 0.1) 1px, transparent 1px),
                linear-gradient(90deg, rgba(100, 255, 218, 0.1) 1px, transparent 1px);
            background-size: 50px 50px;
        }
    </style>
</head>
<body>
    <div class="cover-container">
        <div class="grid-overlay"></div>
        
        <div class="neural-network">
            <!-- Neural network nodes -->
            <div class="node large" style="top: 80px; left: 150px; animation-delay: 0s;"></div>
            <div class="node" style="top: 120px; left: 200px; animation-delay: 0.5s;"></div>
            <div class="node medium" style="top: 60px; left: 250px; animation-delay: 1s;"></div>
            <div class="node" style="top: 180px; left: 180px; animation-delay: 1.5s;"></div>
            <div class="node large" style="top: 140px; left: 300px; animation-delay: 2s;"></div>
            <div class="node" style="top: 100px; left: 350px; animation-delay: 0.3s;"></div>
            <div class="node medium" style="top: 200px; left: 320px; animation-delay: 0.8s;"></div>
            <div class="node" style="top: 90px; left: 400px; animation-delay: 1.3s;"></div>
            <div class="node large" style="top: 160px; left: 450px; animation-delay: 1.8s;"></div>
            
            <!-- Connections -->
            <div class="connection" style="top: 84px; left: 158px; width: 42px; transform: rotate(25deg); animation-delay: 0s;"></div>
            <div class="connection" style="top: 124px; left: 208px; width: 50px; transform: rotate(-15deg); animation-delay: 0.5s;"></div>
            <div class="connection" style="top: 64px; left: 258px; width: 92px; transform: rotate(45deg); animation-delay: 1s;"></div>
            <div class="connection" style="top: 144px; left: 308px; width: 42px; transform: rotate(-30deg); animation-delay: 1.5s;"></div>
            <div class="connection" style="top: 104px; left: 358px; width: 42px; transform: rotate(60deg); animation-delay: 2s;"></div>
        </div>
        
        <!-- Brain outline SVG -->
        <div class="brain-outline">
            <svg width="200" height="150" viewBox="0 0 200 150" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M50 75C50 45 70 25 100 25C130 25 150 45 150 75C150 85 145 90 140 95C150 100 155 110 150 120C145 130 135 135 125 135C115 135 110 130 105 125C100 130 95 135 85 135C75 135 65 130 60 120C55 110 60 100 70 95C65 90 50 85 50 75Z" stroke="#64ffda" stroke-width="2" fill="none"/>
                <circle cx="80" cy="65" r="3" fill="#ff6b6b" opacity="0.6"/>
                <circle cx="120" cy="65" r="3" fill="#a8e6cf" opacity="0.6"/>
                <circle cx="100" cy="85" r="2" fill="#64ffda" opacity="0.6"/>
                <path d="M70 85C80 90 90 85 100 90C110 85 120 90 130 85" stroke="#64ffda" stroke-width="1" opacity="0.4"/>
            </svg>
        </div>
        
        <div class="floating-elements">
            <div class="floating-icon" style="top: 30px; left: 800px; animation-delay: 0s;">🧠</div>
            <div class="floating-icon" style="top: 120px; left: 900px; animation-delay: 2s;">🤖</div>
            <div class="floating-icon" style="top: 200px; left: 850px; animation-delay: 4s;">⚡</div>
            <div class="floating-icon" style="top: 80px; left: 950px; animation-delay: 1s;">🔬</div>
        </div>
        
        <div class="github-corner">⭐</div>
        
        <div class="content">
            <h1 class="title">Thinking Beyond Tokens</h1>
            <p class="subtitle">From Brain-Inspired Intelligence to Cognitive Foundations for AGI</p>
            <p class="description">
                A comprehensive research repository exploring the path from statistical pattern recognition to genuine reasoning. 
                Implementing brain-inspired architectures, vision-language models, agentic AI systems, and ethical frameworks 
                for the development of safe and aligned Artificial General Intelligence.
            </p>
            <div class="tech-stack">
                <span class="tech-tag">Cognitive Architectures</span>
                <span class="tech-tag">Vision-Language Models</span>
                <span class="tech-tag">Neural Societies</span>
                <span class="tech-tag">RLHF & Alignment</span>
                <span class="tech-tag">Tree-of-Thoughts</span>
                <span class="tech-tag">Multimodal Reasoning</span>
                <span class="tech-tag">AGI Benchmarks</span>
            </div>
        </div>
        
        <div class="research-badge">
            arXiv:2507.00951
        </div>
    </div>

    <script>
        // Add some dynamic interaction
        document.addEventListener('DOMContentLoaded', function() {
            const nodes = document.querySelectorAll('.node');
            const connections = document.querySelectorAll('.connection');
            
            // Randomize animation delays for more organic feel
            nodes.forEach((node, index) => {
                node.style.animationDelay = `${Math.random() * 3}s`;
            });
            
            connections.forEach((connection, index) => {
                connection.style.animationDelay = `${Math.random() * 2}s`;
            });
            
            // Add hover effects to tech tags
            const techTags = document.querySelectorAll('.tech-tag');
            techTags.forEach(tag => {
                tag.addEventListener('mouseenter', function() {
                    this.style.transform = 'translateY(-3px) scale(1.05)';
                });
                
                tag.addEventListener('mouseleave', function() {
                    this.style.transform = 'translateY(0) scale(1)';
                });
            });
        });
    </script>
</body>
</html>
