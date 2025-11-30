import React from 'react';

// Re-using the minimal Button definition from the main app for correct typing
const Button = React.forwardRef<HTMLButtonElement, React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "default" | "destructive" | "outline" | "secondary" | "ghost", size?: "default" | "sm" | "icon" }>(
  ({ className, variant = "default", size = "default", ...props }, ref) => {
    // NOTE: The implementation details of Button are irrelevant here, just the type definition matters.
    return (
      <button
        ref={ref}
        className={`p-2 rounded-full text-lg hover:bg-gray-100 ${className || ''}`}
        {...props}
      >
        {props.children}
      </button>
    );
  }
);
Button.displayName = "Button";


interface SimpleModalProps {
  isOpen: boolean;
  onClose: () => void;
  children: React.ReactNode;
  title: string;
}

const SimpleModal: React.FC<SimpleModalProps> = ({ isOpen, onClose, children, title }) => {
  if (!isOpen) return null;

  // The modal content needs to stop propagation to prevent closing when clicking inside the modal
  const handleModalClick = (e: React.MouseEvent) => {
    e.stopPropagation();
  };

  return (
    // Backdrop - Clicking the backdrop closes the modal
    <div 
      className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4 backdrop-blur-sm"
      onClick={onClose}
    >
      {/* Modal Content - Clicks here do NOT close the modal */}
      <div 
        className="bg-background rounded-lg shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto transform scale-95 animate-in fade-in zoom-in duration-300"
        onClick={handleModalClick}
      >
        <div className="flex justify-between items-center p-4 border-b">
          <h3 className="text-xl font-bold">{title}</h3>
          <Button variant="ghost" size="icon" onClick={onClose} aria-label="Close">
            &times; {/* Close X button */}
          </Button>
        </div>
        <div className="p-0"> {/* Adjusted padding to 0 for the FeedbackSection's inner padding */}
          {children}
        </div>
      </div>
    </div>
  );
};

export default SimpleModal;