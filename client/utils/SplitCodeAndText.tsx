export interface MessageSegment {
    type: 'text' | 'code';
    content: string;
    language?: string;  // only present if type is 'code'
}

export function splitMixedContent(content: string): MessageSegment[] {
    const segments: MessageSegment[] = [];
    const parts = content.split(/```/);  // split by triple backticks

    for (let i = 0; i < parts.length; i++) {
        if (i % 2 === 0) {
            // Text block (even index)
            const text = parts[i].trim();
            if (text) {
                segments.push({
                    type: 'text',
                    content: text
                });
            }
        } else {
            // Code block (odd index)
            const codeLines = parts[i].split('\n');
            let language = 'plaintext';
            let codeBody = parts[i];

            if (codeLines.length > 1) {
                language = codeLines[0].trim();
                codeBody = codeLines.slice(1).join('\n').trim();
            }

            segments.push({
                type: 'code',
                language: language || 'plaintext',
                content: codeBody
            });
        }
    }

    return segments;
}