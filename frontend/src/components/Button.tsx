import {FC} from "react";

interface ButtonProps {
	onClick: ()=>void,
	disabled?: boolean,
	label: string
}

export const Button: FC<ButtonProps> = ({onClick, disabled, label}) => {
	return <button
		className={`text-white bg-blue-500 w-full h-10 rounded-xl ${disabled ? "cursor-not-allowed" : "cursor-pointer"}`}
		onClick={onClick}
		disabled={disabled === true}
	>
		{label}
	</button>
}