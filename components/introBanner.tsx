import React, { useState, useEffect } from 'react';
import { Handshake } from 'lucide-react';

// --- Card and Button are assumed to be available ---
// NOTE: I'm cleaning up the 'any' type on Button to meet standard best practices.
type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
    variant?: 'default' | 'primary'; // Assuming a common prop like variant
};

const Card = ({ className, children }: React.HTMLAttributes<HTMLDivElement>) => (
    <div className={`rounded-xl border bg-card text-card-foreground shadow ${className || ""}`}>{children}</div>
);
const Button = ({ children, ...props }: ButtonProps) => <button {...props}>{children}</button>;
// ---------------------------------------------------

interface IntroBannerProps {
    /** The main title to display. */
    title: string;
    /** A brief subtitle or description. */
    subtitle: string;
    /** Optional: The duration (in ms) before the banner automatically fades out. Default is 3500ms (3.5s). */
    autoDismissDuration?: number;
}

export default function IntroBanner({ title, subtitle, autoDismissDuration = 3500 }: IntroBannerProps) {
    // State 1: Controls the initial fade-in transition
    const [isVisible, setIsVisible] = useState(false);
    // State 2: Controls the fade-out/dismissal transition
    const [isDismissed, setIsDismissed] = useState(false);
    // State 3: Controls whether the component is rendered in the DOM
    const [isMounted, setIsMounted] = useState(true);

    const TRANSITION_DURATION = 700; // Must match the duration-700 Tailwind class

    // --- 1. POP-IN LOGIC (Fade In) ---
    useEffect(() => {
        // Trigger the fade-in immediately after mount (Pop-in)
        const fadeInTimer = setTimeout(() => setIsVisible(true), 50); 
        return () => clearTimeout(fadeInTimer);
    }, []);

    // --- 2. POP-OUT LOGIC (Timed Dismissal) ---
    useEffect(() => {
        if (!isVisible) return; // Wait for the fade-in to complete

        // Set a timer to start the dismissal after the configured duration
        const dismissTimer = setTimeout(() => {
            handleDismiss();
        }, autoDismissDuration);

        return () => clearTimeout(dismissTimer);
    }, [isVisible, autoDismissDuration]);

    // --- 3. DISMISSAL HANDLER ---
    const handleDismiss = () => {
        // 1. Trigger the fade-out CSS transition
        setIsDismissed(true);
        
        // 2. Wait for the CSS transition (700ms) to complete
        setTimeout(() => {
            // 3. Remove the component from the DOM entirely (Pop-out)
            setIsMounted(false);
            console.log("Intro Banner dismissed.");
            
            // NOTE: Add logic here if you need to start a process in the parent component
            // (e.g., enable the main UI)
        }, TRANSITION_DURATION);
    };

    // If the component is completely unmounted, stop rendering
    if (!isMounted) {
        return null;
    }

    // Combine logic for Fade-In (isVisible) and Fade-Out (isDismissed)
    const transitionClasses = `
        transition-all **duration-700 ease-in-out**
        ${isDismissed
            ? 'opacity-0 translate-y-4' // FADE-OUT: Hidden and slides down slightly
            : isVisible
                ? 'opacity-100 translate-y-0' // ACTIVE: Fully visible
                : 'opacity-0 -translate-y-4'} // FADE-IN START: Hidden and slides up slightly
    `;

    return (
        <Card 
            className={`p-6 md:p-8 bg-primary/10 border-primary shadow-lg w-full max-w-[1400px] ${transitionClasses}`}
        >
            <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                
                {/* Title and Subtitle */}
                <div className="flex items-center gap-4">
                    <Handshake className="w-8 h-8 md:w-10 md:h-10 text-primary shrink-0" />
                    <div>
                        <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight text-primary">
                            {title}
                        </h1>
                        <p className="text-sm md:text-base text-muted-foreground mt-1">
                            {subtitle}
                        </p>
                    </div>
                </div>

                {/* Optional: Add a call-to-action button */}
                <Button 
                    variant="default" 
                    className="min-w-[120px] shadow-md hover:shadow-lg"
                    // If the user clicks, dismiss the banner immediately
                    onClick={handleDismiss} 
                >
                    Start Training
                </Button>
            </div>
        </Card>
    );
}