import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from threading import Thread
from paraphraser import ParaphraserEngine, SemanticValidator
from ai_avoider import AIDetectionAvoider


class ParaphraserApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Paraphraser & Humanizer")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        
        self.engine = ParaphraserEngine()
        self.avoider = AIDetectionAvoider()
        self.validator = SemanticValidator()
        self.processing = False
        self.original_text_content = ""
        
        self.create_ui()
    
    def create_ui(self):
        """Create the user interface"""
        # Title
        title_frame = ttk.Frame(self.root)
        title_frame.pack(pady=10, padx=10, fill=tk.X)
        
        title = ttk.Label(title_frame, text="Paraphraser & Humanizer", font=("Arial", 16, "bold"))
        title.pack()
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_simple_tab()
        self.create_advanced_tab()
        self.create_help_tab()
        
        # Footer
        footer = ttk.Label(self.root, text="© 2024 Paraphraser - Local Application Only", font=("Arial", 8))
        footer.pack(pady=5)
    
    def create_simple_tab(self):
        """Create the simple paraphrase tab"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Simple Paraphrase")
        
        # Input section
        ttk.Label(frame, text="Original Text:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(10, 0))
        
        self.input_text = scrolledtext.ScrolledText(frame, height=8, wrap=tk.WORD, font=("Arial", 10))
        self.input_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Intensity control
        intensity_frame = ttk.Frame(frame)
        intensity_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(intensity_frame, text="Intensity:").pack(side=tk.LEFT)
        
        # Create label first before slider command
        self.intensity_label = ttk.Label(intensity_frame, text="60%", width=5)
        
        self.intensity_slider = ttk.Scale(intensity_frame, from_=1, to=10, orient=tk.HORIZONTAL, command=self.update_intensity_label)
        self.intensity_slider.set(6)
        self.intensity_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.intensity_label.pack(side=tk.LEFT)
        
        # Humanize checkbox
        self.humanize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Humanize (AI Detection Avoidance)", variable=self.humanize_var).pack(anchor=tk.W, padx=10, pady=5)
        
        # Paraphrase button
        paraphrase_btn = ttk.Button(frame, text="Paraphrase Text", command=self.paraphrase_text)
        paraphrase_btn.pack(fill=tk.X, padx=10, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(frame, variable=self.progress_var, maximum=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)
        
        # Output section
        ttk.Label(frame, text="Paraphrased Text:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(10, 0))
        
        self.output_text = scrolledtext.ScrolledText(frame, height=8, wrap=tk.WORD, font=("Arial", 10))
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Button frame
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(button_frame, text="Copy to Clipboard", command=self.copy_to_clipboard).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Clear All", command=self.clear_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Save Output", command=self.save_output).pack(side=tk.LEFT, padx=2)
    
    def create_advanced_tab(self):
        """Create the advanced options tab"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Advanced Options")
        
        # Techniques frame
        tech_frame = ttk.LabelFrame(frame, text="Techniques", padding=10)
        tech_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.syn_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tech_frame, text="Synonym Replacement", variable=self.syn_var).pack(anchor=tk.W)
        
        self.gram_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tech_frame, text="Grammatical Variation", variable=self.gram_var).pack(anchor=tk.W)
        
        self.struct_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tech_frame, text="Sentence Restructuring", variable=self.struct_var).pack(anchor=tk.W)
        
        # Info frame
        info_frame = ttk.LabelFrame(frame, text="About This Tool", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        info_text = """
This paraphraser uses advanced NLP techniques:

• Synonym Replacement: Replaces words with contextually appropriate synonyms
• Grammatical Variation: Adds/removes contractions and varies word forms
• Sentence Restructuring: Reorganizes sentence structures while maintaining meaning

Intensity slider controls how aggressively text is paraphrased.
Higher intensity = more significant changes.

All processing is done locally - no data sent to any server.
        """
        
        info_label = ttk.Label(info_frame, text=info_text, justify=tk.LEFT)
        info_label.pack(anchor=tk.W)
    
    def create_help_tab(self):
        """Create the help tab"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Help")
        
        help_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=("Arial", 10))
        help_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        help_content = """PARAPHRASER & HUMANIZER - HELP GUIDE

QUICK START:
1. Paste text in the "Original Text" box
2. Adjust the intensity slider (1-10)
3. Keep "Humanize" checked for best AI avoidance
4. Click "Paraphrase Text"
5. Copy or save the output

INTENSITY LEVELS:
• Low (1-3): Subtle changes
• Medium (4-7): Balanced approach (RECOMMENDED)
• High (8-10): Aggressive paraphrasing

TECHNIQUES:
• Synonym Replacement: Changes words to synonyms
• Grammatical Variation: Modifies contractions
• Sentence Restructuring: Reorganizes sentences
• Humanization: Adds human-like patterns

TIPS FOR BEST RESULTS:
✓ Start with medium intensity (5-6)
✓ Keep "Humanize" enabled
✓ Review output carefully
✓ Use multiple paraphrasing rounds for variety
✓ Combine with manual editing for best results

FEATURES:
✓ 100% local processing - no internet needed
✓ No data collection or tracking
✓ Free to use
✓ Works on Windows, Mac, and Linux
✓ Simple, intuitive interface

LIMITATIONS:
⚠ Works best with English text
⚠ Review output for accuracy
⚠ Very technical terms may not be replaced well
⚠ Always verify meaning is preserved

PRIVACY:
All text stays on your computer. No data is sent anywhere.
"""
        
        help_text.insert(tk.END, help_content)
        help_text.config(state=tk.DISABLED)
    
    def update_intensity_label(self, value):
        """Update intensity percentage label"""
        percentage = int(float(value) * 10)
        self.intensity_label.config(text=f"{percentage}%")
    
    def paraphrase_text(self):
        """Paraphrase the input text"""
        text = self.input_text.get("1.0", tk.END).strip()
        
        if not text:
            messagebox.showwarning("Warning", "Please enter some text to paraphrase.")
            return
        
        if self.processing:
            messagebox.showinfo("Info", "Processing already in progress...")
            return
        
        # Run in background thread
        thread = Thread(target=self._process_paraphrase, args=(text,), daemon=True)
        thread.start()
    
    def _process_paraphrase(self, text):
        """Process paraphrasing in background with internal QA improvement"""
        self.processing = True
        self.progress_var.set(0)
        self.original_text_content = text
        
        try:
            intensity = self.intensity_slider.get() / 10.0
            
            # Paraphrase
            self.progress_var.set(30)
            paraphrased = self.engine.paraphrase(text, intensity)
            
            # Humanize if enabled
            self.progress_var.set(60)
            if self.humanize_var.get():
                paraphrased = self.avoider.humanize(paraphrased, intensity)
            
            # Internal QA validation and improvement
            self.progress_var.set(80)
            paraphrased = self.validator.improve_paraphrase(text, paraphrased, self.engine)
            
            self.progress_var.set(100)
            
            # Display result
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert("1.0", paraphrased)
            self.output_text.config(state=tk.NORMAL)
        
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        
        finally:
            self.processing = False
            self.progress_var.set(0)
    
    def copy_to_clipboard(self):
        """Copy output to clipboard"""
        text = self.output_text.get("1.0", tk.END).strip()
        
        if not text:
            messagebox.showwarning("Warning", "There's no text to copy.")
            return
        
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
    
    def clear_all(self):
        """Clear all text"""
        if messagebox.askyesno("Confirm", "Clear all text?"):
            self.input_text.delete("1.0", tk.END)
            self.output_text.delete("1.0", tk.END)
    
    def save_output(self):
        """Save paraphrased text to file"""
        text = self.output_text.get("1.0", tk.END).strip()
        
        if not text:
            messagebox.showwarning("Warning", "There's no text to save.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                messagebox.showinfo("Success", f"File saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {str(e)}")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = ParaphraserApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
