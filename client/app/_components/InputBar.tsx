import { useState, useRef, useEffect } from "react";
import EmojiPicker from "@emoji-mart/react";

type InputBarProps = {
  currentMessage: string;
  setCurrentMessage: React.Dispatch<React.SetStateAction<string>>;
  onSubmit: (e: React.FormEvent<HTMLFormElement>) => void;
};

const InputBar: React.FC<InputBarProps> = ({
  currentMessage,
  setCurrentMessage,
  onSubmit,
}) => {
  const [showEmojiPicker, setShowEmojiPicker] = useState(false);
  const pickerRef = useRef<HTMLDivElement | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setCurrentMessage(e.target.value);
  };

  const handleEmojiSelect = (emoji: any) => {
    setCurrentMessage((prev) => prev + (emoji.native || ""));
  };

  // ✅ Close picker when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        pickerRef.current &&
        !pickerRef.current.contains(event.target as Node)
      ) {
        setShowEmojiPicker(false);
      }
    };

    if (showEmojiPicker) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [showEmojiPicker]);

  return (
    <form onSubmit={onSubmit} className="p-2 md:p-4 bg-white relative">
      <div className="flex items-center bg-[#F9F9F5] rounded-full p-2 md:p-3 shadow-md border border-gray-200">
        <button
          type="button"
          onClick={() => setShowEmojiPicker((prev) => !prev)}
          className="p-2 rounded-full text-gray-500 hover:text-gray-700 hover:bg-gray-100 transition-all duration-200"
        >
          😊
        </button>

        <input
          type="text"
          placeholder="Type a message"
          value={currentMessage}
          onChange={handleChange}
          className="flex-grow px-4 py-2 bg-transparent focus:outline-none text-gray-700"
        />

        <button
          type="submit"
          className="bg-gradient-to-r from-teal-500 to-teal-400 hover:from-teal-600 hover:to-teal-500 rounded-full p-2 md:p-3 md:ml-2 shadow-md transition-all duration-200 group"
        >
          <svg
            className="size-4 md:size-6 text-white transform rotate-45 group-hover:scale-110 transition-transform duration-200"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
            ></path>
          </svg>
        </button>
      </div>

      {showEmojiPicker && (
        <div
          ref={pickerRef}
          className="absolute bottom-16 left-4 z-20"
        >
          <EmojiPicker onEmojiSelect={handleEmojiSelect} />
        </div>
      )}
    </form>
  );
};

export default InputBar;