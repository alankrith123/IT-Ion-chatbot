from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from utils import chunk_text
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Generate a comprehensive response using free text processing
def generate_detailed_response(question, context_chunks):
    """Generate detailed, explanatory responses based on the question and context"""
    # Combine chunks into context
    context = " ".join(context_chunks)
    
    # Clean up the context - remove extra whitespace and format properly
    sentences = [s.strip() for s in context.split('.') if len(s.strip()) > 15]
    question_lower = question.lower()
    
    # Helper function to extract relevant information
    def extract_relevant_info(keywords, max_sentences=5):
        relevant = []
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords):
                relevant.append(sentence.strip())
                if len(relevant) >= max_sentences:
                    break
        return relevant
    
    # Helper function to find the most relevant sentences for any question
    def find_relevant_sentences(question_words, max_sentences=4):
        sentence_scores = []
        for sentence in sentences:
            score = 0
            sentence_lower = sentence.lower()
            for word in question_words:
                if word in sentence_lower:
                    score += 1
            if score > 0:
                sentence_scores.append((sentence, score))
        
        # Sort by relevance score
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        return [sentence for sentence, score in sentence_scores[:max_sentences]]
      # Dynamic response generation based on question type and context
    
    # Contact information questions
    if any(word in question_lower for word in ["contact", "phone", "email", "address", "reach", "call", "write"]):
        contact_keywords = ["contact", "phone", "email", "address", "call", "reach", "location", "@", "+", "tel", "mail"]
        contact_info = extract_relevant_info(contact_keywords, 5)
        
        response = "📞 **Contact Information for IT iON Solutions:**\n\n"
        
        if contact_info:
            response += "**Here's how you can reach us:**\n\n"
            for i, info in enumerate(contact_info, 1):
                response += f"• {info}\n"
            response += "\n"
        else:
            response += "**Get in Touch with IT iON Solutions:**\n\n"
            response += "We'd love to hear from you! Here are the ways to contact us:\n\n"
            response += "📧 **Email:** For general inquiries and business proposals\n"
            response += "📱 **Phone:** Direct line for immediate assistance\n"
            response += "🏢 **Office:** Visit us for in-person consultations\n"
            response += "🌐 **Website:** Complete information about our services\n\n"
        
        response += "**💡 Best Ways to Reach Us:**\n"
        response += "• **For Service Inquiries:** Contact us to discuss your specific IT needs\n"
        response += "• **For Support:** Our team is ready to assist with ongoing projects\n"
        response += "• **For Partnerships:** Let's explore collaboration opportunities\n\n"
        response += "We typically respond within 24 hours and are committed to providing you with the information you need!"
        
        return response
    
    # Company overview questions
    elif any(word in question_lower for word in ["what", "do", "does", "company", "about", "who"]):
        service_keywords = ["solutions", "services", "provides", "offers", "develops", "erp", "sap", "software", "company", "business"]
        relevant_info = extract_relevant_info(service_keywords, 5)
        
        response = "🏢 **About IT iON Solutions:**\n\n"
        
        if relevant_info:
            response += "**Based on our company information:**\n\n"
            for i, info in enumerate(relevant_info[:4], 1):
                response += f"**{i}.** {info}\n\n"
        
        response += "**📋 What We Do:**\n"
        response += "IT iON Solutions is a comprehensive technology company specializing in innovative IT solutions that help businesses optimize their operations and achieve their goals.\n\n"
        
        response += "**🎯 Our Core Focus Areas:**\n"
        response += "• Enterprise Resource Planning (ERP) systems\n"
        response += "• SAP consulting and implementation\n"
        response += "• Custom software development\n"
        response += "• Financial management solutions\n"
        response += "• Business process optimization\n\n"
        
        response += "**💡 Our Approach:** We don't just provide technology - we become a comprehensive part of your enterprise, focusing on long-term partnerships and measurable business impact."
        
        return response
    
    # Services questions
    elif any(word in question_lower for word in ["service", "services", "offer", "provide"]):
        service_keywords = ["service", "erp", "sap", "software", "development", "financial", "presales", "payroll", "retail", "consulting"]
        service_info = extract_relevant_info(service_keywords, 6)
        
        response = "🛠️ **IT iON Solutions Services:**\n\n"
        
        if service_info:
            response += "**Our Key Services Include:**\n\n"
            for i, service in enumerate(service_info[:5], 1):
                response += f"**{i}.** {service}\n\n"
        
        response += "**🔧 Complete Service Portfolio:**\n\n"
        response += "**• iON ERP System** - Streamline your business processes with real-time insights\n"
        response += "**• SAP Implementation** - Full lifecycle consulting and support\n"
        response += "**• Custom Software Development** - Tailored solutions for your specific needs\n"
        response += "**• Financial Management** - Comprehensive oversight and reporting\n"
        response += "**• Business Solutions** - Presales, PayRoll, and Retail management\n\n"
        
        response += "**💼 Why Our Services Work:** All our services integrate seamlessly to provide a comprehensive business solution that grows with your needs."
        
        return response
    
    # Products questions
    elif any(word in question_lower for word in ["product", "products", "erp", "financial", "presales", "payroll", "retail"]):
        product_keywords = ["erp", "financial", "presales", "payroll", "retail", "product", "system", "management", "solution"]
        product_info = extract_relevant_info(product_keywords, 5)
        
        response = "🚀 **IT iON Solutions Products:**\n\n"
        
        if product_info:
            response += "**Our Product Offerings:**\n\n"
            for i, product in enumerate(product_info[:4], 1):
                response += f"**{i}.** {product}\n\n"
        
        response += "**🎯 Main Product Suite:**\n\n"
        response += "**1. iON ERP** - Complete business process management\n"
        response += "**2. iON Financial Management** - Advanced financial oversight\n"
        response += "**3. iON Presales** - Sales pipeline optimization\n"
        response += "**4. iON PayRoll** - Comprehensive payroll solutions\n"
        response += "**5. iON Retail** - Complete retail management\n\n"
        
        response += "**✨ Product Benefits:** Integrated systems that eliminate data silos and provide real-time business insights."
        
        return response
    
    # Why choose us questions
    elif any(word in question_lower for word in ["why", "choose", "advantage", "benefit", "better"]):
        choose_keywords = ["choose", "why", "reliable", "impact", "experience", "client", "customer", "advantage", "better", "best"]
        relevant_info = extract_relevant_info(choose_keywords, 4)
        
        response = "🌟 **Why Choose IT iON Solutions:**\n\n"
        
        if relevant_info:
            response += "**What Sets Us Apart:**\n\n"
            for i, info in enumerate(relevant_info[:3], 1):
                response += f"**{i}.** {info}\n\n"
        
        response += "**🏆 Key Advantages:**\n\n"
        response += "**✅ Proven Track Record** - Reliable services with consistent deliverables\n"
        response += "**✅ True Partnership** - We become part of your enterprise, not just a vendor\n"
        response += "**✅ Comprehensive Solutions** - Full IT suite under one roof\n"
        response += "**✅ Expert Team** - Young, energetic, and experienced professionals\n"
        response += "**✅ Business Impact** - Solutions designed for measurable outcomes\n\n"
        
        response += "**💡 Our Promise:** Technology solutions that transform your business operations and drive real results."
        
        return response
    
    # Team and company culture questions
    elif any(word in question_lower for word in ["team", "culture", "people", "staff", "employee"]):
        team_keywords = ["team", "professional", "experienced", "young", "energetic", "culture", "people", "staff"]
        team_info = extract_relevant_info(team_keywords, 4)
        
        response = "👥 **IT iON Solutions Team:**\n\n"
        
        if team_info:
            response += "**About Our Team:**\n\n"
            for i, info in enumerate(team_info[:3], 1):
                response += f"**{i}.** {info}\n\n"
        
        response += "**🌟 Our People:**\n"
        response += "• **Young & Dynamic** - Fresh perspectives and innovative approaches\n"
        response += "• **Highly Experienced** - Deep expertise across IT domains\n"
        response += "• **Goal-Oriented** - Committed to your business success\n"
        response += "• **Collaborative** - Work as extension of your team\n\n"
        
        response += "**🏢 Company Culture:**\n"
        response += "• Innovation-driven with focus on latest technologies\n"
        response += "• Customer-centric approach prioritizing your needs\n"
        response += "• Quality-first mindset in everything we deliver\n"
        response += "• Long-term partnership philosophy\n\n"
        
        response += "**💼 What This Means for You:** A dedicated team invested in your success and business growth."
        
        return response
    
    # General/fallback response using retrieved context
    else:
        # Use the retrieved context to answer any other questions
        query_words = question_lower.split()
        best_matches = find_relevant_sentences(query_words, 5)
        
        if best_matches:
            response = f"📋 **Answer to: '{question}'**\n\n"
            
            response += "**🎯 Based on our company information:**\n\n"
            for i, sentence in enumerate(best_matches[:4], 1):
                response += f"**{i}.** {sentence}\n\n"
            
            # Add contextual information
            response += "**💡 Additional Context:**\n"
            response += "IT iON Solutions specializes in comprehensive IT solutions including ERP systems, SAP implementation, custom software development, and business process optimization. "
            response += "We focus on providing reliable, scalable technology solutions that drive real business results.\n\n"
            
            response += "**❓ Need More Information?** Feel free to ask about our specific services, products, team, or how we can help your business!"
            
            return response
        else:
            # If no matches found, provide helpful general information
            return f"""🏢 **IT iON Solutions - Here to Help!**

**About Your Question: "{question}"**

While I may not have specific details about that topic, here's what I can tell you about IT iON Solutions:

**🔧 What We Do:**
• Enterprise Resource Planning (ERP) systems
• SAP consulting and implementation  
• Custom software development
• Financial management solutions
• Business process optimization

**🌟 Our Approach:**
We provide comprehensive IT solutions that help businesses streamline operations, improve efficiency, and achieve their goals through innovative technology.

**💬 How We Can Help:**
• Tailored business solutions designed for your needs
• Long-term partnership and reliable support
• Proven track record with growing client base
• Young, experienced team of professionals

**❓ Want to Know More?**
Try asking about:
• Our specific services or products
• Why choose IT iON Solutions
• Our team and company culture  
• Contact information
• How we can help your business

Feel free to ask any other questions about IT iON Solutions!"""

# Load content
with open("company_data.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Chunk it
chunks = chunk_text(raw_text)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Embed chunks
embeddings = model.encode(chunks)
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Initialize summarization pipeline for better responses
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", max_length=100, min_length=30)
    print("✅ Advanced summarization model loaded successfully!")
except:
    print("⚠️ Using basic text processing (advanced model not available)")
    summarizer = None

# Continuous question loop
print("\n" + "="*60)
print("🤖 IT iON Solutions Company Assistant")
print("="*60)
print("Ask me anything about the company! Type 'exit' to quit.")
print("="*60)

while True:
    query = input("\n💬 Your question: ").strip()
    
    if query.lower() in ['exit', 'quit', 'bye']:
        print("\n👋 Thank you for using IT iON Solutions Assistant!")
        break
    
    if not query:
        print("❓ Please ask a question about the company.")
        continue
    
    # Search for relevant chunks
    q_embedding = model.encode([query])
    D, I = index.search(q_embedding, k=5)  # Get more chunks for better context    # Get relevant chunks
    relevant_chunks = [chunks[i] for i in I[0]]
      # Generate and display the response
    print("\n" + "="*60)
    print("📝 DETAILED ANSWER:")
    print("="*60)
    response = generate_detailed_response(query, relevant_chunks)
    print(response)
    print("="*60)
