"use client";

import React, { useState, useRef } from "react";
import emailjs from '@emailjs/browser';
import { MessageSquare, Send, Check, Loader2 } from "lucide-react";

// --- UI Components (Localized here for portability) ---

const Card = ({ className, children }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={`rounded-xl border bg-card text-card-foreground shadow ${className || ""}`}>{children}</div>
);

const Button = React.forwardRef<HTMLButtonElement, React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "default" | "destructive" | "outline" | "secondary" | "ghost", size?: "default" | "sm" | "icon" }>(
  ({ className, variant = "default", ...props }, ref) => {
    const variants = {
      default: "bg-primary text-primary-foreground hover:bg-primary/90",
      destructive: "bg-destructive text-destructive-foreground hover:bg-destructive/90",
      outline: "border border-input bg-background hover:bg-accent hover:text-accent-foreground",
      secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
      ghost: "hover:bg-accent hover:text-accent-foreground"
    };
    return (
      <button ref={ref} className={`inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none disabled:pointer-events-none disabled:opacity-50 h-10 px-4 py-2 ${variants[variant]} ${className || ""}`} {...props} />
    );
  }
);
Button.displayName = "Button";

const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, type, ...props }, ref) => {
    return (
      <input type={type} className={`flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 ${className || ""}`} ref={ref} {...props} />
    )
  }
);
Input.displayName = "Input";

// --- Main Component ---

export default function FeedbackSection() {
  const [status, setStatus] = useState<"idle" | "submitting" | "success" | "error">("idle");
  const formRef = useRef<HTMLFormElement>(null);

  // 🟢 REPLACE WITH YOUR KEYS
  const SERVICE_ID = "service_1fs6sgs";
  const TEMPLATE_ID = "template_zl1jv7p";
  const PUBLIC_KEY = "0cv29_8XIdro2PuMV";

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setStatus("submitting");

    if (!formRef.current) return;

    emailjs.sendForm(SERVICE_ID, TEMPLATE_ID, formRef.current, {
        publicKey: PUBLIC_KEY,
      })
      .then(
        () => {
          setStatus("success");
          formRef.current?.reset();
        },
        (error) => {
          console.error('FAILED...', error.text);
          setStatus("error");
        },
      );
  };

  return (
    <div className="w-full max-w-[1400px] mt-6 animate-in slide-in-from-bottom-4 duration-500">
      <Card className="p-6 md:p-8 border-t-4 border-t-primary">
        <div className="grid md:grid-cols-3 gap-8">
          
          <div className="md:col-span-1 space-y-4">
            <div className="flex items-center gap-2 text-primary font-bold text-xl">
              <MessageSquare className="w-6 h-6" />
              <h2>Feedback & Support</h2>
            </div>
            <p className="text-muted-foreground leading-relaxed">
              Found a bug or have a suggestion? Send us a message directly.
            </p>
          </div>

          <div className="md:col-span-2">
            {status === "success" ? (
              <div className="h-full flex flex-col items-center justify-center text-center p-6 bg-green-500/10 rounded-lg border border-green-500/20">
                <div className="w-12 h-12 bg-green-500 rounded-full flex items-center justify-center mb-4">
                  <Check className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-xl font-bold text-green-700 dark:text-green-400">Message Sent!</h3>
                <Button variant="outline" className="mt-6" onClick={() => setStatus("idle")}>
                  Send Another
                </Button>
              </div>
            ) : (
              <form ref={formRef} onSubmit={handleSubmit} className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Name</label>
                    <Input required name="user_name" placeholder="Your Name" disabled={status === "submitting"} />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Email</label>
                    <Input type="email" required name="user_email" placeholder="name@example.com" disabled={status === "submitting"} />
                  </div>
                </div>
                
                <div className="space-y-2">
                  <label className="text-sm font-medium">Message</label>
                  <textarea 
                    className="flex min-h-[120px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 resize-none"
                    placeholder="Describe the issue or idea..."
                    required
                    name="message"
                    disabled={status === "submitting"}
                  />
                </div>

                <div className="flex justify-end">
                  <Button type="submit" disabled={status === "submitting"} className="min-w-[140px]">
                    {status === "submitting" ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" /> Sending...
                      </>
                    ) : (
                      <>
                        <Send className="w-4 h-4 mr-2" /> Send Feedback
                      </>
                    )}
                  </Button>
                </div>
                
                {status === "error" && (
                  <p className="text-xs text-destructive text-right">
                    Failed to send. Please check your config.
                  </p>
                )}
              </form>
            )}
          </div>
        </div>
      </Card>
    </div>
  );
}