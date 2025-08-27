/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./*.html",
        "../templates/*.html",
        "./*.js",
    ],
    safelist: [
        /* Ensure colors for the comparison table are not dropped. */
        {
            pattern: /c-station\d+/,
        },
        'bg-blue-500',
        'bg-amber-500',
        'bg-emerald-500',
        'bg-violet-500',
    ],
    theme: {
        extend: {},
    },
    plugins: [],
}
